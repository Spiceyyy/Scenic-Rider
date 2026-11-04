"""add-sector: OpenStreetMap -> database, without a GraphML file (download is faked)."""
import osmnx as ox
import pytest
from shapely.geometry import box

from etl import pipeline
from etl.export import export_graph
from etl.extract import extract_from_graph
from etl.load import resolve_sector_ids
from etl.score import run_scoring

from .conftest import NODES, make_graph

WHOLE_AREA = box(-73.585, 45.495, -73.575, 45.505)


def osm_graph(edges=None):
    """What a fresh OSM download looks like: same streets, no scenic scores."""
    G = make_graph()
    for *_, d in G.edges(keys=True, data=True):
        d.pop("scenic_score", None)
        d.pop("scenic_context", None)
    if edges is not None:
        G = G.edge_subgraph(edges).copy()
    return G


@pytest.fixture
def fake_osm(monkeypatch):
    """Map place name -> (graph, boundary) and patch the network calls."""
    places = {}
    monkeypatch.setattr(pipeline, "download_sector",
                        lambda place, network_type="bike": extract_from_graph(places[place][0], f"osm:{place}"))
    monkeypatch.setattr(pipeline, "sector_boundary", lambda place: places[place][1])
    return places


def count(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


def test_add_sector_loads_without_graph_file_and_is_idempotent(db_conn, fake_osm):
    fake_osm["Testville East, Montreal, Canada"] = (osm_graph(), WHOLE_AREA)
    first = pipeline.add_sector(db_conn, "Testville East, Montreal, Canada")
    assert first["load"]["road_segment"]["inserted"] == 4
    assert first["load"]["scenic_score"]["inserted"] == 0          # nothing scored yet
    assert (first["slug"], first["city"], first["new_links"]) == ("testville_east", "Montreal", 4)

    again = pipeline.add_sector(db_conn, "Testville East, Montreal, Canada")
    assert again["load"]["road_segment"] == {"inserted": 0, "updated": 0, "unchanged": 4}
    assert again["new_links"] == 0
    assert count(db_conn, "SELECT count(*) FROM data_source") == 1  # same content -> same source
    assert count(db_conn, "SELECT count(*) FROM sector") == 1


def test_overlapping_sectors_share_segments(db_conn, fake_osm, tmp_path):
    fake_osm["West, Montreal, Canada"] = (osm_graph(), WHOLE_AREA)
    fake_osm["Border, Montreal, Canada"] = (osm_graph([(2, 3, 0), (2, 3, 1)]), WHOLE_AREA)
    pipeline.add_sector(db_conn, "West, Montreal, Canada")
    stats = pipeline.add_sector(db_conn, "Border, Montreal, Canada")

    assert stats["load"]["road_segment"]["inserted"] == 0           # stored once
    assert count(db_conn, "SELECT count(*) FROM road_segment") == 4
    assert count(db_conn, "SELECT count(*) FROM sector_segment WHERE (u, v, key) = (2, 3, 0)") == 2

    with db_conn.cursor() as cur:
        border = resolve_sector_ids(cur, ["border"])
        with pytest.raises(ValueError, match="Known sectors"):
            resolve_sector_ids(cur, ["nowhere"])
    db_conn.rollback()
    info = export_graph(db_conn, tmp_path / "border.graphml", border)
    assert (info["nodes"], info["edges"]) == (2, 2)
    G = ox.load_graphml(tmp_path / "border.graphml")
    assert set(G.edges(keys=True)) == {(2, 3, 0), (2, 3, 1)}


def test_add_sector_keeps_existing_scores(db_conn, fake_osm, graph_file):
    pipeline.run(db_conn, graph_file)                              # scored data from a file
    fake_osm["Same Area, Montreal, Canada"] = (osm_graph(), WHOLE_AREA)
    pipeline.add_sector(db_conn, "Same Area, Montreal, Canada")
    assert count(db_conn, "SELECT count(*) FROM scenic_score") == 3
    assert count(db_conn, "SELECT score FROM scenic_score WHERE (u, v, key) = (1, 2, 0)") == 0.73


def test_link_only_tags_existing_data_by_boundary(db_conn, fake_osm, graph_file):
    pipeline.run(db_conn, graph_file)
    x, y = NODES[1]
    fake_osm["Corner, Montreal, Canada"] = (None, box(x - 0.0002, y - 0.0002, x + 0.0002, y + 0.0002))
    stats = pipeline.add_sector(db_conn, "Corner, Montreal, Canada", link_only=True)
    assert stats["new_links"] == 2                                 # 1->2 and 2->1 touch node 1
    assert count(db_conn, "SELECT osm_downloaded_at IS NULL FROM sector") is True


def test_score_can_be_limited_to_a_sector(db_conn, fake_osm):
    fake_osm["West, Montreal, Canada"] = (osm_graph(), WHOLE_AREA)
    fake_osm["Border, Laval, Canada"] = (osm_graph([(2, 3, 0), (2, 3, 1)]), WHOLE_AREA)
    pipeline.add_sector(db_conn, "West, Montreal, Canada")
    pipeline.add_sector(db_conn, "Border, Laval, Canada")
    with db_conn.cursor() as cur:
        laval = resolve_sector_ids(cur, city="laval")
    db_conn.rollback()
    quiet = dict(sleep=0, log=lambda *a: None, dry_run=True)
    assert run_scoring(db_conn, **quiet)["streets"] == 2               # 1<->2 and 2->3
    assert run_scoring(db_conn, sector_ids=laval, **quiet) == {"streets": 1, "segments": 2}
