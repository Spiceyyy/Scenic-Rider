"""Integration tests against a real PostGIS (skipped unless TEST_DATABASE_URL is set)."""
import networkx as nx
import osmnx as ox
import pytest

from etl import pipeline

from .conftest import make_graph


def _count(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        return cur.fetchone()[0]


def test_initial_load_and_idempotent_rerun(db_conn, graph_file):
    first = pipeline.run(db_conn, graph_file)
    assert first["load"]["road_node"]["inserted"] == 3
    assert first["load"]["road_segment"]["inserted"] == 4
    assert first["load"]["scenic_score"]["inserted"] == 3

    second = pipeline.run(db_conn, graph_file)
    for table in ("road_node", "road_segment", "scenic_score"):
        assert second["load"][table] == {"inserted": 0, "updated": 0, "unchanged": first["load"][table]["inserted"]}
    assert _count(db_conn, "road_segment") == 4
    assert _count(db_conn, "data_source") == 1

    with db_conn.cursor() as cur:
        cur.execute("SELECT ST_SRID(geom), GeometryType(geom) FROM road_segment LIMIT 1")
        assert cur.fetchone() == (4326, "LINESTRING")
        cur.execute("SELECT count(*) FROM etl_run WHERE status = 'success'")
        assert cur.fetchone()[0] == 2


def test_only_changed_rows_are_updated(db_conn, tmp_path):
    path = tmp_path / "g.graphml"
    ox.save_graphml(make_graph(), path)
    pipeline.run(db_conn, path)

    G = make_graph()
    G[1][2][0]["scenic_score"] = 0.91          # one score changed
    G[2][3][1]["scenic_score"] = 0.6           # previously unscored edge now scored
    G[2][3][1]["scenic_context"] = "Mixed / Urban"
    ox.save_graphml(G, path)
    stats = pipeline.run(db_conn, path)

    assert stats["load"]["scenic_score"] == {"inserted": 1, "updated": 1, "unchanged": 2}
    assert stats["load"]["road_segment"] == {"inserted": 0, "updated": 0, "unchanged": 4}
    with db_conn.cursor() as cur:
        cur.execute("SELECT score FROM scenic_score WHERE (u, v, key) = (1, 2, 0)")
        assert cur.fetchone()[0] == pytest.approx(0.91)
        cur.execute("SELECT count(*) FROM data_source")
        assert cur.fetchone()[0] == 2  # new file content -> new source row


def test_strict_mode_rolls_back_and_logs_failure(db_conn, tmp_path):
    G = make_graph()
    G[1][2][0]["scenic_score"] = 3.0  # out of range
    path = tmp_path / "bad.graphml"
    ox.save_graphml(G, path)

    with pytest.raises(pipeline.ValidationFailed):
        pipeline.run(db_conn, path, strict=True)
    assert _count(db_conn, "road_segment") == 0
    with db_conn.cursor() as cur:
        cur.execute("SELECT status FROM etl_run ORDER BY run_id DESC LIMIT 1")
        assert cur.fetchone()[0] == "failed"
