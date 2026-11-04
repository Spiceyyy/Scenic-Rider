"""Load: idempotent UPSERTs into PostGIS.

Change detection: each row carries an md5 `row_hash` of its business columns.
ON CONFLICT ... DO UPDATE only fires when the hash differs, so re-running on an
unchanged file inserts and updates nothing.
"""
import numpy as np
import psycopg2
from psycopg2.extensions import AsIs, register_adapter
from psycopg2.extras import Json, execute_values

from .config import SCHEMA_SQL
from .transform import SCORE_METHODS

PAGE_SIZE = 1000

# pandas hands back numpy scalars; let psycopg2 send them as plain values.
for _t in (np.int32, np.int64):
    register_adapter(_t, lambda x: AsIs(int(x)))
register_adapter(np.float64, lambda x: AsIs(repr(float(x))))
register_adapter(np.bool_, lambda x: AsIs("TRUE" if x else "FALSE"))


def connect(url: str, schema: str):
    conn = psycopg2.connect(url)
    with conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        cur.execute(f'SET search_path TO "{schema}", public')
    conn.commit()
    return conn


def apply_schema(conn):
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    conn.commit()


def register_source(cur, meta: dict) -> int:
    cur.execute(
        """
        INSERT INTO data_source (file_name, file_sha256, created_with, graph_created_at,
                                 source_crs, node_count, edge_count)
        VALUES (%(file_name)s, %(file_sha256)s, %(created_with)s, %(graph_created_at)s,
                %(source_crs)s, %(node_count)s, %(edge_count)s)
        ON CONFLICT (file_sha256) DO UPDATE SET file_name = EXCLUDED.file_name
        RETURNING source_id
        """,
        meta,
    )
    return cur.fetchone()[0]


def seed_methods(cur, model_name: str) -> dict:
    ids = {}
    for code, desc in SCORE_METHODS.items():
        mname = model_name if code == "model" else ""
        cur.execute(
            """
            INSERT INTO score_method (method_code, model_name, description)
            VALUES (%s, %s, %s)
            ON CONFLICT (method_code, model_name) DO UPDATE SET description = EXCLUDED.description
            RETURNING method_id
            """,
            (code, mname, desc),
        )
        ids[code] = cur.fetchone()[0]
    return ids


def _upsert(cur, sql, template, rows) -> dict:
    if not rows:
        return {"inserted": 0, "updated": 0, "unchanged": 0}
    returned = execute_values(cur, sql, rows, template=template, page_size=PAGE_SIZE, fetch=True)
    inserted = sum(1 for (is_new,) in returned if is_new)
    updated = len(returned) - inserted
    return {"inserted": inserted, "updated": updated, "unchanged": len(rows) - len(returned)}


def upsert_nodes(cur, nodes, source_id) -> dict:
    rows = [(r.osmid, r.street_count, r.highway, r.geom_wkb, source_id, r.row_hash)
            for r in nodes.itertuples()]
    return _upsert(cur, """
        INSERT INTO road_node (osmid, street_count, highway, geom, source_id, row_hash)
        VALUES %s
        ON CONFLICT (osmid) DO UPDATE SET
            street_count = EXCLUDED.street_count, highway = EXCLUDED.highway,
            geom = EXCLUDED.geom, source_id = EXCLUDED.source_id,
            row_hash = EXCLUDED.row_hash, updated_at = now()
        WHERE road_node.row_hash IS DISTINCT FROM EXCLUDED.row_hash
        RETURNING (xmax = 0)
        """, "(%s, %s, %s, ST_GeomFromWKB(decode(%s, 'hex'), 4326), %s, %s)", rows)


def upsert_segments(cur, segments, source_id) -> dict:
    rows = [(r["u"], r["v"], r["key"], r["osm_way_ids"], r["name"], r["highway"], r["oneway"],
             r["reversed"], r["length_m"], Json(r["osm_tags"]), r["geom_wkb"], source_id,
             r["row_hash"]) for r in segments.to_dict("records")]
    return _upsert(cur, """
        INSERT INTO road_segment (u, v, key, osm_way_ids, name, highway, oneway, reversed,
                                  length_m, osm_tags, geom, source_id, row_hash)
        VALUES %s
        ON CONFLICT (u, v, key) DO UPDATE SET
            osm_way_ids = EXCLUDED.osm_way_ids, name = EXCLUDED.name,
            highway = EXCLUDED.highway, oneway = EXCLUDED.oneway,
            reversed = EXCLUDED.reversed, length_m = EXCLUDED.length_m,
            osm_tags = EXCLUDED.osm_tags, geom = EXCLUDED.geom,
            source_id = EXCLUDED.source_id, row_hash = EXCLUDED.row_hash, updated_at = now()
        WHERE road_segment.row_hash IS DISTINCT FROM EXCLUDED.row_hash
        RETURNING (xmax = 0)
        """,
        "(%s, %s, %s, %s::bigint[], %s, %s, %s, %s, %s, %s, "
        "ST_GeomFromWKB(decode(%s, 'hex'), 4326), %s, %s)", rows)


def upsert_scores(cur, scores, method_ids: dict, source_id) -> dict:
    rows = [(r["u"], r["v"], r["key"], r["score"], r["context"], r["context_label"],
             method_ids[r["method_code"]], source_id, r["row_hash"])
            for r in scores.to_dict("records")]
    return _upsert(cur, """
        INSERT INTO scenic_score (u, v, key, score, context, context_label, method_id,
                                  source_id, row_hash)
        VALUES %s
        ON CONFLICT (u, v, key) DO UPDATE SET
            score = EXCLUDED.score, context = EXCLUDED.context,
            context_label = EXCLUDED.context_label, method_id = EXCLUDED.method_id,
            source_id = EXCLUDED.source_id, row_hash = EXCLUDED.row_hash, updated_at = now()
        WHERE scenic_score.row_hash IS DISTINCT FROM EXCLUDED.row_hash
           OR scenic_score.method_id IS DISTINCT FROM EXCLUDED.method_id
        RETURNING (xmax = 0)
        """, "(%s, %s, %s, %s, %s, %s, %s, %s, %s)", rows)


def start_run(conn, source_file: str) -> int:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO etl_run (status, source_file) VALUES ('running', %s) RETURNING run_id",
                    (source_file,))
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def finish_run(conn, run_id, status, source_id=None, stats=None, error=None):
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE etl_run SET status = %s, finished_at = now(), source_id = %s,
                      stats = %s, error = %s WHERE run_id = %s""",
            (status, source_id, Json(stats) if stats is not None else None, error, run_id),
        )
    conn.commit()
