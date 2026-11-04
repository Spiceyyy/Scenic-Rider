"""Load green areas (parks, gardens, woods) from OpenStreetMap into PostGIS.

They feed v_segment_score: segments that run through a green area get a bonus,
and the "Inferred (Path)" guess only applies to paths that are mostly inside one.
"""
import time

import osmnx as ox
import requests
import shapely
from osmnx._errors import InsufficientResponseError
from psycopg2.extras import execute_values

GREEN_TAGS = {
    "leisure": ["park", "garden", "nature_reserve"],
    "landuse": ["forest", "recreation_ground"],
    "natural": ["wood"],
}


def _kind(row):
    for tag, values in GREEN_TAGS.items():
        value = row.get(tag)
        if isinstance(value, str) and value in values:
            return f"{tag}={value}"
    return None


def download_green(polygon, attempts=3, wait=10):
    """Green-area polygons intersecting `polygon` -> list of row tuples.
    The public Overpass server sometimes drops connections, so retry with backoff."""
    for attempt in range(1, attempts + 1):
        try:
            gdf = ox.features_from_polygon(polygon, tags=GREEN_TAGS)
            break
        except InsufficientResponseError:  # nothing green in the area
            return []
        except requests.RequestException:
            if attempt == attempts:
                raise
            time.sleep(wait * attempt)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    rows = []
    for (element, osm_id), row in gdf.iterrows():
        kind = _kind(row)
        if kind:
            name = row.get("name")
            rows.append((element, int(osm_id), name if isinstance(name, str) else None, kind,
                         shapely.to_wkb(row.geometry, hex=True)))
    return rows


def upsert_green(cur, rows):
    if not rows:
        return {"inserted": 0, "updated": 0}
    returned = execute_values(cur, """
        INSERT INTO green_area (osm_type, osm_id, name, kind, geom) VALUES %s
        ON CONFLICT (osm_type, osm_id) DO UPDATE SET
            name = EXCLUDED.name, kind = EXCLUDED.kind, geom = EXCLUDED.geom, loaded_at = now()
        WHERE NOT ST_Equals(green_area.geom, EXCLUDED.geom)
           OR green_area.name IS DISTINCT FROM EXCLUDED.name
           OR green_area.kind IS DISTINCT FROM EXCLUDED.kind
        RETURNING (xmax = 0)
        """, rows,
        template="(%s, %s, %s, %s, ST_Multi(ST_CollectionExtract("
                 "ST_MakeValid(ST_GeomFromWKB(decode(%s, 'hex'), 4326)), 3)))",
        page_size=500, fetch=True)
    inserted = sum(1 for (new,) in returned if new)
    return {"inserted": inserted, "updated": len(returned) - inserted}


def load_green(conn, sector_ids=None, download=download_green):
    """Download green areas for each sector boundary (all sectors by default)."""
    with conn.cursor() as cur:
        cur.execute("""SELECT sector_id, slug, ST_AsBinary(boundary) FROM sector
                       WHERE boundary IS NOT NULL AND (%s OR sector_id = ANY(%s))
                       ORDER BY slug""", (sector_ids is None, list(sector_ids or [])))
        sectors = cur.fetchall()
    conn.rollback()
    results = {}
    for sector_id, slug, wkb in sectors:
        rows = download(shapely.from_wkb(bytes(wkb)))
        with conn.cursor() as cur:
            results[slug] = {"found": len(rows), **upsert_green(cur, rows)}
        conn.commit()
    return results
