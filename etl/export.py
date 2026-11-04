"""Export: rebuild an OSMnx GraphML file from PostGIS (database -> graph file).

scenic_score in the file is the FINAL score from v_segment_score (raw score plus
the park bonus / inferred-path rule); scenic_raw_score keeps the original value.

The result has the attributes app.py / find_scenic_route.py rely on: node x/y,
edge length, geometry, scenic_score, scenic_context, plus the other OSM tags.
"""
import ast
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
import shapely

# Selected segments: everything, or only those linked to the requested sectors.
_SELECTED = """
SELECT r.* FROM road_segment r
WHERE %(all_sectors)s OR EXISTS (
    SELECT 1 FROM sector_segment ss
    WHERE ss.sector_id = ANY(%(sector_ids)s) AND (ss.u, ss.v, ss.key) = (r.u, r.v, r.key))
"""
NODES_SQL = f"""
WITH sel AS ({_SELECTED})
SELECT n.osmid, n.street_count, n.highway, ST_AsBinary(n.geom) FROM road_node n
WHERE n.osmid IN (SELECT u FROM sel UNION SELECT v FROM sel)
ORDER BY n.osmid
"""
EDGES_SQL = f"""
WITH sel AS ({_SELECTED})
SELECT r.u, r.v, r.key, r.osm_way_ids, r.name, r.highway, r.oneway, r.reversed,
       r.length_m, r.osm_tags, ST_AsBinary(r.geom),
       f.final_score, f.context, f.method_code, f.raw_score, f.green_share
FROM sel r
JOIN v_segment_score f USING (u, v, key)   -- final score = raw score + rules (see schema.sql)
ORDER BY r.u, r.v, r.key
"""


def _split(text):
    """Undo the ';' join used on load: 'a;b' -> ['a', 'b'], 'a' -> 'a'."""
    if text is None:
        return None
    parts = text.split(";")
    return parts if len(parts) > 1 else text


def _reversed(text):
    """'False' -> False, 'False;True' -> [False, True] (lists were ';'-joined on load)."""
    if text is None:
        return None
    try:
        values = [ast.literal_eval(part) for part in text.split(";")]
    except (ValueError, SyntaxError):
        return text
    return values if len(values) > 1 else values[0]


def export_graph(conn, out_path, sector_ids=None):
    params = {"all_sectors": sector_ids is None, "sector_ids": list(sector_ids or [])}
    with conn.cursor() as cur:
        cur.execute(NODES_SQL, params)
        node_rows = cur.fetchall()
        cur.execute(EDGES_SQL, params)
        edge_rows = cur.fetchall()
    conn.rollback()
    if not node_rows:
        raise RuntimeError("Nothing to export: no road data for that selection. "
                           "Load data with `python -m etl add-sector` or `python -m etl run`.")

    node_geoms = shapely.from_wkb([bytes(r[3]) for r in node_rows])
    nodes = gpd.GeoDataFrame({
        "y": shapely.get_y(node_geoms), "x": shapely.get_x(node_geoms),
        "street_count": [r[1] for r in node_rows],
        "highway": [r[2] for r in node_rows],
    }, geometry=node_geoms, crs="EPSG:4326",
       index=pd.Index([r[0] for r in node_rows], name="osmid"))

    records = []
    for (u, v, key, way_ids, name, highway, oneway, rev, length, tags, wkb,
         score, context, method, raw_score, green_share) in edge_rows:
        rec = dict(tags or {})
        rec.update({
            "u": u, "v": v, "key": key,
            "osmid": (way_ids[0] if way_ids and len(way_ids) == 1 else way_ids),
            "name": _split(name), "highway": _split(highway), "oneway": oneway,
            "reversed": _reversed(rev), "length": length,
            "geometry": shapely.from_wkb(bytes(wkb)),
        })
        if score is not None:  # unscored edges stay unscored (app fills 0.5)
            rec.update(scenic_score=score, scenic_context=context, scenic_method=method,
                       scenic_raw_score=raw_score, green_share=round(green_share, 3))
        records.append(rec)
    edges = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326").set_index(["u", "v", "key"])

    G = ox.graph_from_gdfs(nodes, edges, graph_attrs={"crs": "epsg:4326", "simplified": True,
                                                       "created_with": f"Scenic ETL export (OSMnx {ox.__version__})"})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, out_path)
    return {"nodes": len(G.nodes), "edges": len(G.edges),
            "scored_edges": sum(1 for *_, d in G.edges(data=True) if "scenic_score" in d),
            "path": str(out_path)}
