import pandas as pd
from shapely.geometry import LineString, Point

from etl.extract import extract_graph
from etl.pipeline import prepare
from etl.transform import clean_label, method_for_context, transform_edges, transform_nodes
from etl.validate import validate


def _frames(graph_file):
    ext = extract_graph(graph_file)
    nodes = transform_nodes(ext.nodes)
    segments, scores = transform_edges(ext.edges)
    return ext, nodes, segments, scores


def checks(result):
    return {(i.entity, i.check): i.count for i in result.issues}


def test_extract_and_transform_shapes(graph_file):
    ext, nodes, segments, scores = _frames(graph_file)
    assert ext.meta["node_count"] == 3 and ext.meta["edge_count"] == 4
    assert len(ext.meta["file_sha256"]) == 64
    assert len(segments) == 4 and len(scores) == 3  # parallel edge has no score
    assert set(scores["method_code"]) == {"model", "default", "inferred_tag"}
    assert scores.loc[scores["method_code"] == "model", "context_label"].item() == "Green Tunnel"


def test_hashes_are_deterministic(graph_file):
    _, n1, s1, sc1 = _frames(graph_file)
    _, n2, s2, sc2 = _frames(graph_file)
    assert list(n1.row_hash) == list(n2.row_hash)
    assert list(s1.row_hash) == list(s2.row_hash)
    assert list(sc1.row_hash) == list(sc2.row_hash)


def test_clean_fixture_has_no_errors(graph_file):
    _, result = prepare(graph_file)
    assert result.errors == []
    assert checks(result)[("segment", "segment_without_score")] == 1


def test_method_mapping():
    assert method_for_context("Waterfront 🌊") == "model"
    assert method_for_context("No Data") == "default"
    assert method_for_context("Inferred (Path)") == "inferred_tag"
    assert method_for_context("Error") == "error_fallback"
    assert method_for_context("???") == "unknown"
    assert clean_label("City St (No Bike Lane) 🚗") == "City St (No Bike Lane)"


def test_validation_rejects_bad_rows(graph_file):
    _, nodes, segments, scores = _frames(graph_file)

    nodes = pd.concat([nodes, pd.DataFrame([
        {"osmid": None, "geometry": Point(0, 0)},             # missing id
        {"osmid": 9, "geometry": Point(500, 500)},            # out-of-range coords
    ])], ignore_index=True)
    segments = pd.concat([segments, segments.iloc[[0]], pd.DataFrame([
        {"u": 1, "v": 77, "key": 0, "length_m": 1.0,           # orphan (node 77 missing)
         "geometry": LineString([(0, 0), (0, 1)])},
        {"u": 3, "v": 1, "key": 0, "length_m": 1.0,           # invalid geometry
         "geometry": LineString([(0, 0), (0, 0)])},
    ])], ignore_index=True)
    scores = pd.concat([scores, pd.DataFrame([
        {"u": 2, "v": 1, "key": 5, "score": 0.4, "method_code": "model"},   # orphan score
    ])], ignore_index=True)
    scores.loc[scores["method_code"] == "inferred_tag", "score"] = 1.7      # out of range

    result = validate(nodes, segments, scores)
    c = checks(result)
    assert c[("node", "missing_id")] == 1
    assert c[("node", "invalid_geometry")] == 1
    assert c[("segment", "duplicate_key")] == 2
    assert c[("segment", "orphan_segment")] == 1
    assert c[("segment", "invalid_geometry")] == 1
    assert c[("score", "orphan_score")] >= 1
    assert c[("score", "score_out_of_range")] == 1

    # Only clean rows survive: the duplicated (1,2,0) segment is rejected, so its
    # score becomes orphaned too; remaining = 2->1 and the two 2->3 segments.
    assert len(result.nodes) == 3
    assert sorted(map(tuple, result.segments[["u", "v", "key"]].values.tolist())) == \
        [(2, 1, 0), (2, 3, 0), (2, 3, 1)]
    assert result.scores["score"].between(0, 1).all()
