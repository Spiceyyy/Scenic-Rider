"""Orchestrates extract -> transform -> validate -> load for one GraphML file."""
import logging

from .config import scoring_model_name
from .extract import extract_graph
from .load import (finish_run, register_source, seed_methods, start_run, upsert_nodes,
                   upsert_scores, upsert_segments)
from .transform import transform_edges, transform_nodes
from .validate import validate

log = logging.getLogger("etl")


class ValidationFailed(Exception):
    pass


def prepare(graph_path):
    """Extract, transform and validate (no database needed)."""
    ext = extract_graph(graph_path)
    nodes = transform_nodes(ext.nodes)
    segments, scores = transform_edges(ext.edges)
    return ext.meta, validate(nodes, segments, scores)


def run(conn, graph_path, strict=False) -> dict:
    run_id = start_run(conn, str(graph_path))
    source_id = None
    try:
        meta, result = prepare(graph_path)
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
        conn.commit()
        finish_run(conn, run_id, "success", source_id, stats)
        return stats
    except Exception as exc:
        conn.rollback()
        finish_run(conn, run_id, "failed", None, None, f"{type(exc).__name__}: {exc}")
        raise
