"""Validate: data-quality checks on transformed records.

Rows failing an ERROR check are excluded from the load (and reported); WARN/INFO
checks are reported only. Use `--strict` on the CLI to abort on any ERROR.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

ERROR, WARN, INFO = "error", "warn", "info"
LON_RANGE, LAT_RANGE = (-180.0, 180.0), (-90.0, 90.0)


@dataclass
class Issue:
    check: str
    entity: str
    severity: str
    count: int
    examples: list = field(default_factory=list)


@dataclass
class ValidationResult:
    nodes: pd.DataFrame
    segments: pd.DataFrame
    scores: pd.DataFrame
    issues: list

    @property
    def errors(self):
        return [i for i in self.issues if i.severity == ERROR]

    def summary(self) -> dict:
        return {
            "rows_valid": {"nodes": len(self.nodes), "segments": len(self.segments),
                           "scores": len(self.scores)},
            "issues": [asdict(i) for i in self.issues],
        }


def _examples(df, cols, n=5):
    return [list(map(lambda x: x.item() if isinstance(x, np.generic) else x, r))
            for r in df[cols].head(n).itertuples(index=False)]


def _geom_problems(geoms, expected_type):
    """Boolean mask: null/empty/wrong type/invalid/out-of-range coordinates."""
    bad = []
    for g in geoms:
        if g is None or g.is_empty or g.geom_type != expected_type or not g.is_valid:
            bad.append(True)
            continue
        minx, miny, maxx, maxy = g.bounds
        bad.append(not (LON_RANGE[0] <= minx <= maxx <= LON_RANGE[1]
                        and LAT_RANGE[0] <= miny <= maxy <= LAT_RANGE[1]))
    return pd.Series(bad, index=geoms.index, dtype=bool)


def validate(nodes: pd.DataFrame, segments: pd.DataFrame, scores: pd.DataFrame) -> ValidationResult:
    issues = []

    def flag(mask, df, check, entity, cols, severity=ERROR):
        mask = mask.astype(bool)
        n = int(mask.sum())
        if n:
            issues.append(Issue(check, entity, severity, n, _examples(df[mask], cols)))
        return mask

    # --- nodes -----------------------------------------------------------------
    bad_n = flag(nodes["osmid"].isna(), nodes, "missing_id", "node", ["osmid"])
    bad_n |= flag(nodes["osmid"].duplicated(keep=False) & nodes["osmid"].notna(),
                  nodes, "duplicate_key", "node", ["osmid"])
    bad_n |= flag(_geom_problems(nodes["geometry"], "Point"), nodes,
                  "invalid_geometry", "node", ["osmid"])
    nodes_ok = nodes[~bad_n]
    valid_node_ids = set(nodes_ok["osmid"])

    # --- segments --------------------------------------------------------------
    k = ["u", "v", "key"]
    bad_s = flag(segments[k].isna().any(axis=1), segments, "missing_id", "segment", k)
    bad_s |= flag(segments.duplicated(k, keep=False) & segments[k].notna().all(axis=1),
                  segments, "duplicate_key", "segment", k)
    bad_s |= flag(_geom_problems(segments["geometry"], "LineString"), segments,
                  "invalid_geometry", "segment", k)
    bad_s |= flag(segments["length_m"].isna() | (segments["length_m"] < 0), segments,
                  "invalid_length", "segment", k + ["length_m"])
    bad_s |= flag(~segments["u"].isin(valid_node_ids) | ~segments["v"].isin(valid_node_ids),
                  segments, "orphan_segment", "segment", k)
    flag(segments["osm_way_ids"].isna(), segments, "missing_osm_way_id", "segment", k, WARN)
    segs_ok = segments[~bad_s]

    # --- scores ----------------------------------------------------------------
    valid_seg_keys = pd.MultiIndex.from_frame(segs_ok[k])
    score_keys = pd.MultiIndex.from_frame(scores[k])
    bad_sc = flag(scores[k].isna().any(axis=1), scores, "missing_id", "score", k)
    bad_sc |= flag(scores.duplicated(k, keep=False), scores, "duplicate_key", "score", k)
    bad_sc |= flag(~pd.Series(score_keys.isin(valid_seg_keys), index=scores.index),
                   scores, "orphan_score", "score", k)
    s = pd.to_numeric(scores["score"], errors="coerce")
    bad_sc |= flag(~np.isfinite(s) | (s < 0) | (s > 1), scores,
                   "score_out_of_range", "score", k + ["score_raw"])
    flag(scores["method_code"] == "unknown", scores, "unknown_context", "score",
         k + ["context"], WARN)
    scores_ok = scores[~bad_sc]

    # --- coverage (informational) ---------------------------------------------
    unscored = ~pd.Series(valid_seg_keys.isin(pd.MultiIndex.from_frame(scores_ok[k])),
                          index=segs_ok.index)
    flag(unscored, segs_ok, "segment_without_score", "segment", k, INFO)
    flag(scores_ok["method_code"] != "model", scores_ok, "score_not_from_model", "score",
         k + ["method_code"], INFO)

    return ValidationResult(nodes_ok, segs_ok, scores_ok, issues)
