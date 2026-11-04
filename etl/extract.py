"""Extract: read an existing scored OSMnx GraphML file into GeoDataFrames."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import osmnx as ox


@dataclass
class Extracted:
    nodes: gpd.GeoDataFrame  # index: osmid
    edges: gpd.GeoDataFrame  # index: (u, v, key)
    meta: dict


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_graph(path) -> Extracted:
    path = Path(path)
    G = ox.load_graphml(path)  # same loader app.py uses -> same typing rules
    nodes, edges = ox.graph_to_gdfs(G)  # fills straight-line geometry where missing
    meta = {
        "file_name": path.name,
        "file_sha256": file_sha256(path),
        "created_with": G.graph.get("created_with"),
        "graph_created_at": G.graph.get("created_date"),
        "source_crs": str(G.graph.get("crs", "")),
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
    return Extracted(nodes=nodes, edges=edges, meta=meta)
