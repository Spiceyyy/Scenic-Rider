"""Database-first scoring and export (needs TEST_DATABASE_URL; Mapillary is faked)."""
import networkx as nx
import osmnx as ox
import pytest

from etl import pipeline
from etl.export import export_graph
from etl.score import run_scoring

from .conftest import make_graph

QUIET = dict(sleep=0, log=lambda *a: None)


def fake_fetch(status, score=None):
    calls = []

    def fetch(lat, lon):
        calls.append((lat, lon))
        r = {"status": status, "score": score, "context": None, "image_id": None,
             "captured_at": None, "image_distance_m": None, "http_status": 200, "error": None}
        if status == "found":
            r.update(context="Mixed / Urban", image_id="img-42", image_distance_m=7.5)
        return r
    fetch.calls = calls
    return fetch


def scores(conn):
    with conn.cursor() as cur:
        cur.execute("""SELECT u, v, key, score, m.method_code, s.image_id
                       FROM scenic_score s JOIN score_method m USING (method_id)""")
        return {(u, v, k): (sc, m, img) for u, v, k, sc, m, img in cur.fetchall()}


def test_found_scores_both_directions_with_one_search(db_conn, graph_file):
    pipeline.run(db_conn, graph_file)
    fetch = fake_fetch("found", 0.66)
    counts = run_scoring(db_conn, fetch=fetch, **QUIET)

    # Targets: 2->1 (default), 2->3/0 (inferred), 2->3/1 (unscored). 2->3 key 0 and 1
    # share a geometry, so they are one street: 2 searches, 3 segments updated.
    assert len(fetch.calls) == 2 and counts["segments_updated"] == 3
    s = scores(db_conn)
    assert s[(1, 2, 0)] == (0.73, "model", None)            # existing model score untouched
    for k in [(2, 1, 0), (2, 3, 0), (2, 3, 1)]:
        assert s[k] == (0.66, "model", "img-42")

    # Nothing left to do on the next run.
    assert run_scoring(db_conn, fetch=fetch, dry_run=True, **QUIET)["streets"] == 0

    # Re-loading the old graph file must not overwrite the image-based scores.
    pipeline.run(db_conn, graph_file)
    assert scores(db_conn)[(2, 1, 0)] == (0.66, "model", "img-42")


def test_api_error_changes_nothing_and_is_retried(db_conn, graph_file):
    pipeline.run(db_conn, graph_file)
    before = scores(db_conn)
    run_scoring(db_conn, fetch=fake_fetch("api_error"), **QUIET)
    assert scores(db_conn) == before
    with db_conn.cursor() as cur:
        cur.execute("SELECT status, count(*) FROM score_attempt GROUP BY status")
        assert dict(cur.fetchall()) == {"api_error": 2}
    assert run_scoring(db_conn, fetch=fake_fetch("found", 0.6), dry_run=True, **QUIET)["streets"] == 2


def test_no_images_fills_only_gaps_and_waits_before_retry(db_conn, graph_file):
    pipeline.run(db_conn, graph_file)
    fetch = fake_fetch("no_images")
    run_scoring(db_conn, fetch=fetch, **QUIET)
    # Straight 2-point streets are searched once (midpoint only), like build_graph.py.
    assert len(fetch.calls) == 2
    s = scores(db_conn)
    assert s[(2, 1, 0)] == (0.5, "default", None)           # kept
    assert s[(2, 3, 0)] == (0.8, "inferred_tag", None)      # kept
    assert s[(2, 3, 1)] == (0.5, "default", None)           # gap filled with the old rule
    assert run_scoring(db_conn, fetch=fetch, dry_run=True, **QUIET)["streets"] == 0
    assert run_scoring(db_conn, fetch=fetch, dry_run=True, retry_days=0, **QUIET)["streets"] == 2


def test_stops_after_repeated_failures(db_conn, graph_file):
    pipeline.run(db_conn, graph_file)
    with pytest.raises(RuntimeError, match="in a row"):
        run_scoring(db_conn, fetch=fake_fetch("api_error"), max_consecutive_failures=1, **QUIET)
    with db_conn.cursor() as cur:
        cur.execute("SELECT status FROM etl_run WHERE job = 'score'")
        assert cur.fetchone()[0] == "failed"


def _routes(G, pairs):
    for _, _, _, d in G.edges(keys=True, data=True):
        length = float(d.get("length", 1.0))
        d["scenic_cost"] = length / max(float(d.get("scenic_score", 0.5)), 0.05)
    return [(nx.shortest_path(G, a, b, weight="length"),
             nx.shortest_path(G, a, b, weight="scenic_cost")) for a, b in pairs]


def test_export_round_trip_matches_original(db_conn, graph_file, tmp_path):
    pipeline.run(db_conn, graph_file)
    out = tmp_path / "export.graphml"
    info = export_graph(db_conn, out)
    assert (info["nodes"], info["edges"], info["scored_edges"]) == (3, 4, 3)

    original, exported = ox.load_graphml(graph_file), ox.load_graphml(out)
    assert set(original.edges(keys=True)) == set(exported.edges(keys=True))
    for u, v, k, d in original.edges(keys=True, data=True):
        e = exported[u][v][k]
        assert e["length"] == d["length"]
        assert "scenic_score" in d or "scenic_score" not in e
        if d.get("scenic_context") == "Inferred (Path)":
            # no green area loaded -> the 0.8 path guess falls back to neutral 0.5
            assert (e["scenic_score"], e["scenic_raw_score"]) == ("0.5", d["scenic_score"])
        else:
            assert e.get("scenic_score") == d.get("scenic_score")
    for _, _, d in original.edges(data=True):  # same rule applied to the reference graph
        if d.get("scenic_context") == "Inferred (Path)":
            d["scenic_score"] = "0.5"
    pairs = [(1, 3), (3, 1), (2, 1)] if nx.has_path(original, 3, 1) else [(1, 3), (2, 1)]
    assert _routes(original, pairs) == _routes(exported, pairs)
