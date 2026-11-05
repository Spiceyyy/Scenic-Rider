"""request.py with the network and the model mocked (no API calls, no weights loaded)."""
import io
import sys
import types

import pytest
from PIL import Image


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b""):
        self.status_code, self._json, self.content = status_code, json_data, content

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


def _jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "green").save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def req(monkeypatch):
    monkeypatch.setenv("MAPILLARY_TOKEN", "test-token")
    fake_model = types.ModuleType("calculate_score")
    fake_model.calculate_score = lambda path, debug=False: (0.66, "Green Tunnel 🌳")
    monkeypatch.setitem(sys.modules, "calculate_score", fake_model)
    import request
    monkeypatch.setattr(request, "BACKOFF_SECONDS", 0)
    return request


def _queue(monkeypatch, req, responses):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(req.requests, "get", fake_get)
    return calls


IMAGES = {"data": [
    {"id": "1", "thumb_1024_url": "http://img/1", "captured_at": 1700000000000, "is_pano": False,
     "geometry": {"type": "Point", "coordinates": [-73.5800, 45.5000]}},
    {"id": "2", "thumb_1024_url": "http://img/2", "captured_at": 1600000000000, "is_pano": True,
     "geometry": {"type": "Point", "coordinates": [-73.5801, 45.5001]}},
]}


def test_no_images_is_not_an_error(monkeypatch, req):
    _queue(monkeypatch, req, [FakeResponse(200, {"data": []})])
    assert req.search_images(45.5, -73.58)["status"] == "no_images"


def test_rate_limit_is_retried_then_succeeds(monkeypatch, req):
    calls = _queue(monkeypatch, req, [FakeResponse(429, {}), FakeResponse(200, IMAGES)])
    result = req.search_images(45.5, -73.58)
    assert result["status"] == "found" and len(calls) == 2
    assert result["image"]["id"] == "2"  # panorama preferred, as before


def test_persistent_failure_is_api_error(monkeypatch, req):
    _queue(monkeypatch, req, [FakeResponse(500, {})] * (req.MAX_RETRIES + 1))
    result = req.search_images(45.5, -73.58)
    assert result["status"] == "api_error" and "500" in result["error"]


def test_bad_token_is_api_error_without_retry(monkeypatch, req):
    calls = _queue(monkeypatch, req, [FakeResponse(401, {"error": {"message": "Invalid token"}})])
    result = req.search_images(45.5, -73.58)
    assert (result["status"], result["http_status"], len(calls)) == ("api_error", 401, 1)


def test_network_exception_is_api_error(monkeypatch, req):
    _queue(monkeypatch, req, [req.requests.ConnectionError("down")] * (req.MAX_RETRIES + 1))
    assert req.search_images(45.5, -73.58)["status"] == "api_error"


def test_fetch_and_score_returns_provenance(monkeypatch, req):
    _queue(monkeypatch, req, [FakeResponse(200, IMAGES), FakeResponse(200, content=_jpeg_bytes())])
    r = req.fetch_and_score(45.5, -73.58)
    assert r["status"] == "found" and r["score"] == 0.66 and r["image_id"] == "2"
    assert r["captured_at"].year == 2020
    assert 0 < r["image_distance_m"] < 20


def test_download_failure(monkeypatch, req):
    _queue(monkeypatch, req, [FakeResponse(200, IMAGES)] + [FakeResponse(503)] * (req.MAX_RETRIES + 1))
    assert req.fetch_and_score(45.5, -73.58)["status"] == "download_error"


def test_legacy_interface_unchanged(monkeypatch, req):
    _queue(monkeypatch, req, [FakeResponse(200, IMAGES), FakeResponse(200, content=_jpeg_bytes())])
    assert req.download_and_score(45.5, -73.58) == (0.66, "Green Tunnel 🌳")
    _queue(monkeypatch, req, [FakeResponse(200, {"data": []})])
    assert req.download_and_score(45.5, -73.58) == (None, "No Data")
    _queue(monkeypatch, req, [FakeResponse(500)] * (req.MAX_RETRIES + 1))
    assert req.download_and_score(45.5, -73.58) == (None, "API Error")
