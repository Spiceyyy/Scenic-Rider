"""Transform: normalise OSMnx GeoDataFrames into flat records matching db/schema.sql."""
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd
import shapely

from .config import TARGET_CRS

# Labels emitted by calculate_score.calculate_score (vision model output).
MODEL_CONTEXT_PREFIXES = ("Waterfront", "Dedicated Bike Lane", "Green Tunnel",
                          "City St", "Mixed / Urban", "Neutral")

# score_method rows: method_code -> description. 'model' additionally carries model_name.
SCORE_METHODS = {
    "model": "Mask2Former semantic segmentation of a Mapillary image (calculate_score.py)",
    "inferred_tag": "No image found; inferred 0.8 from OSM highway tag (build_graph.py)",
    "default": "No image found; neutral default 0.5 (build_graph.py)",
    "error_fallback": "Image could not be opened; calculate_score returned 0.5",
    "unknown": "Context label not recognised by the ETL",
}

SEGMENT_COLUMNS = {"osmid", "name", "highway", "oneway", "reversed", "length",
                   "geometry", "scenic_score", "scenic_context"}


def method_for_context(context) -> str:
    if not isinstance(context, str) or not context:
        return "unknown"
    if context == "No Data":
        return "default"
    if context.startswith("Inferred"):
        return "inferred_tag"
    if context in ("Error", "Download Error"):
        return "error_fallback"
    if context.startswith(MODEL_CONTEXT_PREFIXES):
        return "model"
    return "unknown"


def clean_label(context):
    if not isinstance(context, str):
        return None
    return re.sub(r"[^\x00-\x7F]+", "", context).strip() or None


def _py(value):
    """numpy / NaN -> plain JSON-serialisable Python values."""
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _join(value):
    value = _py(value)
    if value is None:
        return None
    if isinstance(value, list):
        return ";".join(str(v) for v in value)
    return str(value)


def _way_ids(value):
    value = _py(value)
    if value is None:
        return None
    items = value if isinstance(value, list) else [value]
    try:
        return [int(v) for v in items]
    except (TypeError, ValueError):
        return None


def _to_int(value):
    value = _py(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_score(raw):
    """Return (float or None, is_present). Non-numeric present values -> (nan, True)."""
    raw = _py(raw)
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return None, False
    try:
        return float(raw), True
    except (TypeError, ValueError):
        return float("nan"), True


def row_hash(values: dict) -> str:
    payload = json.dumps(values, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _wkb_hex(geoms):
    return [None if g is None or g.is_empty else shapely.to_wkb(g, hex=True) for g in geoms]


def _to_target_crs(gdf):
    if gdf.crs is None:
        return gdf.set_crs(TARGET_CRS)
    if gdf.crs.to_epsg() != 4326:
        return gdf.to_crs(TARGET_CRS)
    return gdf


def transform_nodes(nodes) -> pd.DataFrame:
    nodes = _to_target_crs(nodes)
    df = pd.DataFrame({
        "osmid": [_to_int(i) for i in nodes.index],
        "street_count": [_to_int(v) for v in nodes.get("street_count", pd.Series(index=nodes.index))],
        "highway": [_join(v) for v in nodes.get("highway", pd.Series(index=nodes.index))],
        "geometry": list(nodes.geometry),
    })
    df["geom_wkb"] = _wkb_hex(df["geometry"])
    df["row_hash"] = [
        row_hash({"street_count": r.street_count, "highway": r.highway, "geom": r.geom_wkb})
        for r in df.itertuples()
    ]
    return df


def transform_edges(edges):
    """Return (segments_df, scores_df). Every edge yields a segment row; only edges
    that carry a scenic_score attribute yield a score row."""
    edges = _to_target_crs(edges)
    idx = edges.index.to_frame(index=False)
    idx.columns = ["u", "v", "key"]
    tag_cols = [c for c in edges.columns if c not in SEGMENT_COLUMNS]

    seg_rows, score_rows = [], []
    for (u, v, key), (_, e) in zip(idx.itertuples(index=False), edges.iterrows()):
        u, v, key = _to_int(u), _to_int(v), _to_int(key)
        geom = e.geometry
        tags = {c: _py(e[c]) for c in tag_cols if _py(e[c]) is not None}
        length = _py(e.get("length"))
        seg = {
            "u": u, "v": v, "key": key,
            "osm_way_ids": _way_ids(e.get("osmid")),
            "name": _join(e.get("name")),
            "highway": _join(e.get("highway")),
            "oneway": _py(e.get("oneway")),
            "reversed": _join(e.get("reversed")),
            "length_m": float(length) if length is not None else None,
            "osm_tags": tags,
            "geometry": geom,
        }
        seg_rows.append(seg)

        score, present = parse_score(e.get("scenic_score"))
        if present:
            context = _py(e.get("scenic_context"))
            score_rows.append({
                "u": u, "v": v, "key": key,
                "score": score,
                "score_raw": _py(e.get("scenic_score")),
                "context": context,
                "context_label": clean_label(context),
                "method_code": method_for_context(context),
            })

    segments = pd.DataFrame(seg_rows, columns=[
        "u", "v", "key", "osm_way_ids", "name", "highway", "oneway", "reversed",
        "length_m", "osm_tags", "geometry"])
    segments["geom_wkb"] = _wkb_hex(segments["geometry"])
    segments["row_hash"] = [
        row_hash({k: r[k] for k in ("osm_way_ids", "name", "highway", "oneway",
                                    "reversed", "length_m", "osm_tags", "geom_wkb")})
        for r in segments.to_dict("records")
    ]

    scores = pd.DataFrame(score_rows, columns=[
        "u", "v", "key", "score", "score_raw", "context", "context_label", "method_code"])
    scores["row_hash"] = [
        row_hash({"score": r["score"], "context": r["context"], "method": r["method_code"]})
        for r in scores.to_dict("records")
    ]
    return segments, scores
