"""Tiny synthetic fixtures (test-only; not project results)."""
import os
import sys
import uuid
from pathlib import Path

import networkx as nx
import osmnx as ox
import pytest
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NODES = {1: (-73.580, 45.500), 2: (-73.579, 45.501), 3: (-73.578, 45.502)}


def make_graph():
    G = nx.MultiDiGraph(crs="epsg:4326", created_with="OSMnx 2.0.7", created_date="test")
    for nid, (x, y) in NODES.items():
        G.add_node(nid, x=x, y=y, street_count=2)

    def edge(u, v, key=0, **attrs):
        geom = LineString([NODES[u], NODES[v]])
        G.add_edge(u, v, key=key, osmid=100 + u, length=140.0, highway="residential",
                   oneway=False, reversed=False, geometry=geom, **attrs)

    edge(1, 2, scenic_score=0.73, scenic_context="Green Tunnel 🌳")
    edge(2, 1, scenic_score=0.5, scenic_context="No Data")
    edge(2, 3, scenic_score=0.8, scenic_context="Inferred (Path)")
    edge(2, 3, key=1)  # parallel edge left unscored, like the real graph
    return G


@pytest.fixture
def graph_file(tmp_path):
    path = tmp_path / "tiny.graphml"
    ox.save_graphml(make_graph(), path)
    return path


@pytest.fixture
def db_conn():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    from etl.load import apply_schema, connect

    schema = f"scenic_test_{uuid.uuid4().hex[:8]}"
    conn = connect(url, schema)
    apply_schema(conn)
    yield conn
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f'DROP SCHEMA "{schema}" CASCADE')
    conn.commit()
    conn.close()
