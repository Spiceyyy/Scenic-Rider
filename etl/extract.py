"""Extract: get an OSMnx road graph, either from an existing GraphML file or
downloaded straight from OpenStreetMap for a named sector."""
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import osmnx as ox


@dataclass
class Extracted:
    nodes: gpd.GeoDataFrame  # index: osmid
    edges: gpd.GeoDataFrame  # index: (u, v, key)
    meta: dict               # file_sha256 may be None -> filled from content hashes


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_from_graph(G, file_name, sha256=None) -> Extracted:
    nodes, edges = ox.graph_to_gdfs(G)  # fills straight-line geometry where missing
    meta = {
        "file_name": file_name,
        "file_sha256": sha256,
        "created_with": G.graph.get("created_with"),
        "graph_created_at": G.graph.get("created_date"),
        "source_crs": str(G.graph.get("crs", "")),
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
    return Extracted(nodes=nodes, edges=edges, meta=meta)


def extract_graph(path) -> Extracted:
    path = Path(path)
    G = ox.load_graphml(path)  # same loader app.py uses -> same typing rules
    return extract_from_graph(G, path.name, file_sha256(path))


def sector_boundary(place):
    """Geocode a place name to its boundary polygon (EPSG:4326)."""
    gdf = ox.geocode_to_gdf(place)
    return gdf.to_crs("EPSG:4326").geometry.union_all()


def download_sector(place, network_type="bike") -> Extracted:
    """Same OSMnx call build_graph.py makes, but kept in memory (no GraphML file).

    truncate_by_edge=True keeps streets that cross the sector boundary (with their
    outside intersection), so neighbouring sectors share border intersections and
    routes can cross from one sector into the next.
    """
    G = ox.graph_from_place(place, network_type=network_type, truncate_by_edge=True)
    G.graph.setdefault("created_date", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return extract_from_graph(G, f"osm:{place}")
