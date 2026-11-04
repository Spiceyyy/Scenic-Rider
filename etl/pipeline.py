"""Orchestrates extract -> transform -> validate -> load.

Two entry points share the same transform/validate/load path:
  run(conn, graph_path)        existing GraphML file -> database
  add_sector(conn, place, ...) OpenStreetMap download -> database (no file)
"""
import hashlib
import logging
import re

import shapely
from psycopg2.extras import execute_values

from .config import scoring_model_name
from .extract import download_sector, extract_graph, sector_boundary
from .load import (finish_run, register_source, seed_methods, start_run, upsert_nodes,
                   upsert_scores, upsert_segments)
from .transform import transform_edges, transform_nodes
from .validate import validate

log = logging.getLogger("etl")


class ValidationFailed(Exception):
    pass


def prepare_extracted(ext):
    nodes = transform_nodes(ext.nodes)
    segments, scores = transform_edges(ext.edges)
    if ext.meta.get("file_sha256") is None:
        # Downloaded data has no file: identify it by its content instead.
        h = hashlib.sha256()
        for value in sorted(nodes["row_hash"]) + sorted(segments["row_hash"]):
            h.update(value.encode())
        ext.meta["file_sha256"] = h.hexdigest()
    return ext.meta, validate(nodes, segments, scores)


def prepare(graph_path):
    """Extract, transform and validate a GraphML file (no database needed)."""
    return prepare_extracted(extract_graph(graph_path))


def _load(conn, run_id, meta, result, strict, after_load=None):
    stats = {"run_id": run_id, "source": meta, "validation": result.summary()}
    if strict and result.errors:
        raise ValidationFailed(f"{len(result.errors)} error check(s) failed (strict mode)")

    # Single transaction for all data: either everything lands or nothing does.
    with conn.cursor() as cur:
        source_id = register_source(cur, meta)
        method_ids = seed_methods(cur, scoring_model_name())
        stats["load"] = {
            "road_node": upsert_nodes(cur, result.nodes, source_id),
            "road_segment": upsert_segments(cur, result.segments, source_id),
            "scenic_score": upsert_scores(cur, result.scores, method_ids, source_id),
        }
        if after_load:
            after_load(cur, stats)
    conn.commit()
    finish_run(conn, run_id, "success", source_id, stats)
    return stats


def run(conn, graph_path, strict=False) -> dict:
    run_id = start_run(conn, str(graph_path))
    try:
        meta, result = prepare(graph_path)
        return _load(conn, run_id, meta, result, strict)
    except Exception as exc:
        conn.rollback()
        finish_run(conn, run_id, "failed", None, None, f"{type(exc).__name__}: {exc}")
        raise


# --------------------------------------------------------------------------- sectors

def sector_slug(place):
    """'Le Sud-Ouest, Montreal, Canada' -> 'le_sud_ouest' (same rule as build_graph.py)."""
    slug = re.sub(r"[^a-z0-9]", "_", place.split(",")[0].strip().lower())
    return re.sub(r"_+", "_", slug).strip("_")


def default_city(place):
    parts = [p.strip() for p in place.split(",") if p.strip()]
    return parts[1] if len(parts) >= 2 else parts[0]


def _upsert_sector(cur, place, city, network_type, boundary, downloaded):
    cur.execute(
        """
        INSERT INTO sector (name, slug, city, network_type, boundary, osm_downloaded_at)
        VALUES (%s, %s, %s, %s, ST_Multi(ST_GeomFromWKB(decode(%s, 'hex'), 4326)),
                CASE WHEN %s THEN now() END)
        ON CONFLICT (name) DO UPDATE SET
            city = EXCLUDED.city, network_type = EXCLUDED.network_type,
            boundary = EXCLUDED.boundary,
            osm_downloaded_at = COALESCE(EXCLUDED.osm_downloaded_at, sector.osm_downloaded_at),
            updated_at = now()
        RETURNING sector_id
        """,
        (place, sector_slug(place), city, network_type,
         shapely.to_wkb(boundary, hex=True), downloaded),
    )
    return cur.fetchone()[0]


def _sector_summary(cur, sector_id):
    cur.execute(
        """SELECT count(*),
                  count(s.score),
                  count(*) FILTER (WHERE m.method_code = 'model')
           FROM sector_segment ss
           LEFT JOIN scenic_score s USING (u, v, key)
           LEFT JOIN score_method m USING (method_id)
           WHERE ss.sector_id = %s""", (sector_id,))
    total, scored, model = cur.fetchone()
    return {"segments": total, "scored": scored, "model_scored": model}


def add_sector(conn, place, city=None, network_type="bike", strict=False, link_only=False):
    """Download a sector from OpenStreetMap into the database and record which
    segments belong to it.

    link_only=True skips the download and only tags segments already in the
    database that intersect the sector boundary (used for data loaded earlier
    from GraphML files, so it is not re-downloaded and changed).
    """
    city = city or default_city(place)
    boundary = sector_boundary(place)

    if link_only:
        with conn.cursor() as cur:
            sector_id = _upsert_sector(cur, place, city, network_type, boundary, False)
            cur.execute(
                """INSERT INTO sector_segment (sector_id, u, v, key)
                   SELECT %s, r.u, r.v, r.key FROM road_segment r JOIN sector s
                     ON s.sector_id = %s AND ST_Intersects(r.geom, s.boundary)
                   ON CONFLICT DO NOTHING""", (sector_id, sector_id))
            new_links = cur.rowcount
            summary = _sector_summary(cur, sector_id)
        conn.commit()
        return {"sector_id": sector_id, "slug": sector_slug(place), "city": city,
                "new_links": new_links, "sector": summary}

    run_id = start_run(conn, f"osm:{place}", job="add_sector")
    try:
        meta, result = prepare_extracted(download_sector(place, network_type))

        def link(cur, stats):
            sector_id = _upsert_sector(cur, place, city, network_type, boundary, True)
            rows = [(sector_id, int(u), int(v), int(k))
                    for u, v, k in result.segments[["u", "v", "key"]].itertuples(index=False)]
            returned = execute_values(
                cur, "INSERT INTO sector_segment (sector_id, u, v, key) VALUES %s "
                     "ON CONFLICT DO NOTHING RETURNING 1", rows, page_size=1000, fetch=True)
            stats.update(sector_id=sector_id, slug=sector_slug(place), city=city,
                         new_links=len(returned), sector=_sector_summary(cur, sector_id))

        return _load(conn, run_id, meta, result, strict, after_load=link)
    except Exception as exc:
        conn.rollback()
        finish_run(conn, run_id, "failed", None, None, f"{type(exc).__name__}: {exc}")
        raise
