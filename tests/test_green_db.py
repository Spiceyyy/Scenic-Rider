"""Park bonus, inferred-path rule (v_segment_score), green loading, legacy re-scoring."""
import pytest
import shapely
from shapely.geometry import box

from etl import pipeline
from etl.green import load_green
from etl.score import run_scoring

from .test_score_export_db import fake_fetch, scores

# Covers the 2->3 street completely and only touches node 2 (a corner) of 1<->2.
PARK = box(-73.5790, 45.5010, -73.5775, 45.5025)
QUIET = dict(sleep=0, log=lambda *a: None)


def add_park(conn, geom=PARK, osm_id=1):
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO green_area (osm_type, osm_id, name, kind, geom)
                       VALUES ('way', %s, 'Test Park', 'leisure=park',
                               ST_Multi(ST_GeomFromText(%s, 4326)))""", (osm_id, geom.wkt))
    conn.commit()


def final_scores(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT u, v, key, round(green_share::numeric, 2), base_score, final_score "
                    "FROM v_segment_score")
        return {(u, v, k): (float(share), base, final) for u, v, k, share, base, final in cur.fetchall()}


def test_inferred_guess_needs_a_park(db_conn, graph_file):
    pipeline.run(db_conn, graph_file)
    f = final_scores(db_conn)
    assert f[(2, 3, 0)] == (0.0, 0.5, 0.5)           # 0.8 guess, but not in a park -> 0.5
    assert f[(1, 2, 0)] == (0.0, 0.73, 0.73)         # model score unchanged
    assert f[(2, 3, 1)] == (0.0, None, None)         # unscored stays unscored


def test_park_bonus(db_conn, graph_file):
    pipeline.run(db_conn, graph_file)
    add_park(db_conn)
    f = final_scores(db_conn)
    assert f[(2, 3, 0)] == (1.0, 0.8, pytest.approx(0.95))   # guess kept + full bonus
    assert f[(2, 3, 1)] == (1.0, None, pytest.approx(0.65))  # unscored, in park: 0.5 + bonus
    assert f[(1, 2, 0)] == (0.0, 0.73, 0.73)                 # only touches the park at a point
    assert scores(db_conn)[(2, 3, 0)][0] == 0.8              # raw data never modified


def test_load_green_is_idempotent(db_conn, graph_file):
    with db_conn.cursor() as cur:
        cur.execute("""INSERT INTO sector (name, slug, city, boundary) VALUES
                       ('T, Montreal', 't', 'Montreal', ST_Multi(ST_GeomFromText(%s, 4326)))""",
                    (box(-73.6, 45.4, -73.5, 45.6).wkt,))
    db_conn.commit()
    rows = [("way", 7, "Parc Test", "leisure=park", shapely.to_wkb(PARK, hex=True))]
    assert load_green(db_conn, download=lambda poly: rows)["t"] == {"found": 1, "inserted": 1, "updated": 0}
    assert load_green(db_conn, download=lambda poly: rows)["t"] == {"found": 1, "inserted": 0, "updated": 0}


def test_rescore_legacy_and_bbox(db_conn, graph_file):
    pipeline.run(db_conn, graph_file)
    fetch = fake_fetch("found", 0.93)
    # (1,2,0) is a legacy model score (no photo): skipped by default ...
    run_scoring(db_conn, fetch=fetch, **QUIET)
    assert scores(db_conn)[(1, 2, 0)] == (0.73, "model", None)
    # ... re-scored with rescore_legacy, but only inside the bbox
    far_away = (-70.0, 40.0, -69.9, 40.1)
    assert run_scoring(db_conn, fetch=fetch, rescore_legacy=True, bbox=far_away,
                       dry_run=True, **QUIET)["streets"] == 0
    run_scoring(db_conn, fetch=fetch, rescore_legacy=True, bbox=(-73.6, 45.4, -73.5, 45.6), **QUIET)
    assert scores(db_conn)[(1, 2, 0)] == (0.93, "model", "img-42")
    # scores from a recorded photo are never re-scored
    assert run_scoring(db_conn, fetch=fetch, rescore_legacy=True, dry_run=True, **QUIET)["streets"] == 0
