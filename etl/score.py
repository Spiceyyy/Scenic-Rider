"""Database-first scoring: pick streets that lack an image-based score, look them up
on Mapillary via request.py, and write the result straight into PostGIS.

Rules
* A "street" is all segments with the same geometry regardless of direction, so a
  two-way street (A->B and B->A) costs one Mapillary search, not two.
* Only segments without a model score are targeted (unscored, default, inferred,
  error fallback). Existing model scores are never touched.
* api_error / download_error / score_error never change a score; they are only
  logged in score_attempt so the street is retried next run.
* no_images: existing fallback scores are kept; unscored segments get the same
  fallback build_graph.py uses (0.8 for cycleway/path/footway, else 0.5).
  The street is skipped for --retry-after-days before being searched again.
* Each street is committed separately, so an interrupted run loses nothing.
"""
import sys
import time

from .config import PROJECT_ROOT, scoring_model_name
from .load import finish_run, seed_methods, start_run
from .transform import clean_label, row_hash

CHECKPOINTS = (0.5, 0.25, 0.75)          # same order build_graph.py uses
PATH_HIGHWAYS = {"cycleway", "path", "footway"}
FAILURE_STATUSES = {"api_error", "download_error", "score_error"}

# Direction-independent geometry: A->B and B->A map to the same line.
_CANON = "CASE WHEN r.u <= r.v THEN r.geom ELSE ST_Reverse(r.geom) END"

TARGETS_SQL = f"""
WITH target AS (
    SELECT r.u, r.v, r.key, coalesce(r.highway, '') AS highway,
           s.score IS NULL AS unscored,
           {_CANON} AS cgeom,
           md5(ST_AsBinary({_CANON})) AS street_hash
    FROM road_segment r
    LEFT JOIN scenic_score s USING (u, v, key)
    LEFT JOIN score_method m USING (method_id)
    WHERE (s.score IS NULL OR m.method_code <> 'model')
      AND (%(all_sectors)s OR EXISTS (
            SELECT 1 FROM sector_segment ss
            WHERE ss.sector_id = ANY(%(sector_ids)s)
              AND (ss.u, ss.v, ss.key) = (r.u, r.v, r.key)))
), street AS (
    SELECT street_hash,
           array_agg(u ORDER BY u, v, key)        AS us,
           array_agg(v ORDER BY u, v, key)        AS vs,
           array_agg(key ORDER BY u, v, key)      AS keys,
           array_agg(highway ORDER BY u, v, key)  AS highways,
           array_agg(unscored ORDER BY u, v, key) AS unscored,
           (array_agg(cgeom))[1]                  AS geom
    FROM target
    GROUP BY street_hash
)
SELECT st.street_hash, st.us, st.vs, st.keys, st.highways, st.unscored,
       ST_NPoints(st.geom) AS npoints,
       ST_Y(ST_LineInterpolatePoint(st.geom, 0.5)),  ST_X(ST_LineInterpolatePoint(st.geom, 0.5)),
       ST_Y(ST_LineInterpolatePoint(st.geom, 0.25)), ST_X(ST_LineInterpolatePoint(st.geom, 0.25)),
       ST_Y(ST_LineInterpolatePoint(st.geom, 0.75)), ST_X(ST_LineInterpolatePoint(st.geom, 0.75))
FROM street st
WHERE NOT EXISTS (
    SELECT 1 FROM score_attempt a
    WHERE a.street_hash = st.street_hash AND a.status = 'no_images'
      AND a.attempted_at > now() - make_interval(days => %(retry_days)s))
ORDER BY st.street_hash
LIMIT %(limit)s
"""

UPSERT_SCORE_SQL = """
INSERT INTO scenic_score (u, v, key, score, context, context_label, method_id, source_id,
                          row_hash, image_id, image_captured_at, image_distance_m, scored_run_id)
VALUES (%(u)s, %(v)s, %(key)s, %(score)s, %(context)s, %(context_label)s, %(method_id)s, NULL,
        %(row_hash)s, %(image_id)s, %(captured_at)s, %(image_distance_m)s, %(run_id)s)
ON CONFLICT (u, v, key) DO UPDATE SET
    score = EXCLUDED.score, context = EXCLUDED.context, context_label = EXCLUDED.context_label,
    method_id = EXCLUDED.method_id, source_id = NULL, row_hash = EXCLUDED.row_hash,
    image_id = EXCLUDED.image_id, image_captured_at = EXCLUDED.image_captured_at,
    image_distance_m = EXCLUDED.image_distance_m, scored_run_id = EXCLUDED.scored_run_id,
    updated_at = now()
"""


def default_fetch(lat, lon):
    """Real lookup via request.py (imported lazily: it loads the model on first score)."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from request import fetch_and_score
    return fetch_and_score(lat, lon)


def find_targets(cur, limit, retry_days, sector_ids=None):
    cur.execute(TARGETS_SQL, {"limit": limit, "retry_days": retry_days,
                              "all_sectors": sector_ids is None,
                              "sector_ids": list(sector_ids or [])})
    streets = []
    for row in cur.fetchall():
        (street_hash, us, vs, keys, highways, unscored, npoints, *coords) = row
        points = list(zip(CHECKPOINTS, coords[0::2], coords[1::2]))
        if npoints <= 2:          # straight line: build_graph.py only checks the middle
            points = points[:1]
        streets.append({
            "street_hash": street_hash,
            "segments": [dict(u=u, v=v, key=k, highway=h, unscored=n)
                         for u, v, k, h, n in zip(us, vs, keys, highways, unscored)],
            "points": points,
        })
    return streets


def _write_score(cur, seg, score, context, method_code, method_ids, run_id, prov=None):
    prov = prov or {}
    cur.execute(UPSERT_SCORE_SQL, {
        **seg, "score": float(score), "context": context, "context_label": clean_label(context),
        "method_id": method_ids[method_code],
        "row_hash": row_hash({"score": float(score), "context": context, "method": method_code}),
        "image_id": prov.get("image_id"), "captured_at": prov.get("captured_at"),
        "image_distance_m": prov.get("image_distance_m"), "run_id": run_id,
    })


def _log_attempt(cur, run_id, street_hash, result, checkpoint, updated):
    cur.execute(
        """INSERT INTO score_attempt (run_id, street_hash, status, checkpoint, http_status,
                                      image_id, error, segments_updated)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (run_id, street_hash, result["status"], checkpoint, result.get("http_status"),
         result.get("image_id"), result.get("error"), updated))


def score_street(cur, street, fetch, method_ids, run_id):
    """Search the street's checkpoints; write and log the outcome. Returns status."""
    result, checkpoint = None, None
    for checkpoint, lat, lon in street["points"]:
        result = fetch(lat, lon)
        if result["status"] != "no_images":
            break                 # found, or a failure: stop searching this street

    status = result["status"]
    updated = 0
    if status == "found":
        for seg in street["segments"]:
            _write_score(cur, seg, result["score"], result["context"], "model",
                         method_ids, run_id, result)
            updated += 1
    elif status == "no_images":
        for seg in street["segments"]:
            if seg["unscored"]:   # keep existing fallbacks; fill only true gaps
                inferred = seg["highway"] in PATH_HIGHWAYS
                _write_score(cur, seg, 0.8 if inferred else 0.5,
                             "Inferred (Path)" if inferred else "No Data",
                             "inferred_tag" if inferred else "default", method_ids, run_id)
                updated += 1
    _log_attempt(cur, run_id, street["street_hash"], result, checkpoint, updated)
    return status, updated


def run_scoring(conn, limit=50, retry_days=30, fetch=None, sleep=0.5,
                max_consecutive_failures=5, dry_run=False, log=print, sector_ids=None):
    fetch = fetch or default_fetch
    with conn.cursor() as cur:
        streets = find_targets(cur, limit, retry_days, sector_ids)
    conn.rollback()
    if dry_run:
        segs = sum(len(s["segments"]) for s in streets)
        log(f"Dry run: {len(streets)} street(s) / {segs} segment(s) would be searched.")
        return {"streets": len(streets), "segments": segs}

    run_id = start_run(conn, "mapillary", job="score")
    counts = {"streets": 0, "segments_updated": 0}
    consecutive_failures = 0
    try:
        with conn.cursor() as cur:
            method_ids = seed_methods(cur, scoring_model_name())
        conn.commit()

        for i, street in enumerate(streets, 1):
            with conn.cursor() as cur:
                status, updated = score_street(cur, street, fetch, method_ids, run_id)
            conn.commit()         # one transaction per street -> safe to interrupt
            counts["streets"] += 1
            counts[status] = counts.get(status, 0) + 1
            counts["segments_updated"] += updated
            log(f"[{i}/{len(streets)}] {status:14} segments updated: {updated}")

            consecutive_failures = consecutive_failures + 1 if status in FAILURE_STATUSES else 0
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(f"{consecutive_failures} failures in a row (last: {status}); "
                                   "stopping. Check the token / rate limit and re-run.")
            if sleep and i < len(streets):
                time.sleep(sleep)
        finish_run(conn, run_id, "success", stats=counts)
        return counts
    except BaseException as exc:  # includes Ctrl+C: completed streets stay committed
        conn.rollback()
        finish_run(conn, run_id, "failed", stats=counts, error=f"{type(exc).__name__}: {exc}")
        raise
