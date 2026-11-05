"""Mapillary image lookup + scenic scoring for one coordinate.

Two entry points:
  * fetch_and_score(lat, lon) -> dict with an explicit status and image provenance.
    Used by the database scorer (`python -m etl score`).
  * download_and_score(lat, lon, save_image=False) -> (score, context).
    Original interface, still used by build_graph.py (behaviour unchanged).

Statuses (so "no photo exists" is never confused with "the request failed"):
  found          image found and scored by the model
  no_images      Mapillary answered successfully but has no image nearby
  api_error      request failed after retries (rate limit, server error, bad token...)
  download_error image metadata found but the image file could not be downloaded
  score_error    image downloaded but the model could not read it
"""
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from io import BytesIO

import requests
from PIL import Image

# --- CONFIGURATION ---
# The token is read from the MAPILLARY_TOKEN environment variable (put it in .env).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

SEARCH_URL = "https://graph.mapillary.com/images"
MIN_DATE = "2020-01-01"
SEARCH_DELTA_DEG = 0.0005      # half-width of the search box (~55 m north-south)
MAX_RETRIES = 3                # extra attempts after the first one
BACKOFF_SECONDS = 2.0          # waits 2s, 4s, 8s between retries
TIMEOUT_SECONDS = 30
RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def _token():
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise RuntimeError("MAPILLARY_TOKEN is not set. Add it to your .env file.")
    return token


def _get_with_retries(url, params=None):
    """GET with retry/backoff. Returns (response or None, error message or None)."""
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        if attempt:
            time.sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
            continue
        if response.status_code in RETRYABLE_HTTP:
            last_error = f"HTTP {response.status_code}"
            continue
        return response, None
    return None, last_error


def _distance_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (haversine)."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def search_images(lat, lon):
    """Find the best Mapillary image near a coordinate.

    Returns a dict: status ('found' | 'no_images' | 'api_error'), image (or None),
    http_status, error.
    """
    delta = SEARCH_DELTA_DEG
    params = {
        "access_token": _token(),
        "fields": "id,thumb_1024_url,captured_at,is_pano,geometry",
        "bbox": f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}",
        "start_captured_at": f"{MIN_DATE}T00:00:00Z",
        "limit": 10,
    }
    response, error = _get_with_retries(SEARCH_URL, params)
    if response is None:
        return {"status": "api_error", "image": None, "http_status": None, "error": error}

    try:
        data = response.json()
    except ValueError:
        return {"status": "api_error", "image": None, "http_status": response.status_code,
                "error": "response was not JSON"}
    if response.status_code != 200 or "error" in data:
        message = (data.get("error") or {}).get("message", "") if isinstance(data, dict) else ""
        return {"status": "api_error", "image": None, "http_status": response.status_code,
                "error": message or f"HTTP {response.status_code}"}

    candidates = data.get("data", [])
    if not candidates:
        return {"status": "no_images", "image": None, "http_status": 200, "error": None}

    # Same preference as before: panoramas first, then the most recent capture.
    best = max(candidates, key=lambda img: (1 if img.get("is_pano") else 0,
                                            img.get("captured_at") or 0))
    return {"status": "found", "image": best, "http_status": 200, "error": None}


def get_best_image_at_coordinate(lat, lon):
    """Original helper: thumbnail URL of the best image, or None."""
    result = search_images(lat, lon)
    if result["status"] == "api_error":
        print(f"API Error: {result['error']}")
    return result["image"]["thumb_1024_url"] if result["image"] else None


def _image_provenance(image, lat, lon):
    captured = image.get("captured_at")
    coords = (image.get("geometry") or {}).get("coordinates")
    return {
        "image_id": str(image["id"]) if image.get("id") is not None else None,
        "captured_at": (datetime.fromtimestamp(captured / 1000, tz=timezone.utc)
                        if isinstance(captured, (int, float)) else None),
        "image_distance_m": (_distance_m(lat, lon, coords[1], coords[0])
                             if coords and len(coords) == 2 else None),
    }


def fetch_and_score(lat, lon, save_image=False):
    """Search, download and score. Never raises for network/data problems; the
    outcome is in result['status']."""
    found = search_images(lat, lon)
    result = {"status": found["status"], "score": None, "context": None,
              "image_id": None, "captured_at": None, "image_distance_m": None,
              "http_status": found["http_status"], "error": found["error"]}
    if found["status"] != "found":
        return result
    result.update(_image_provenance(found["image"], lat, lon))

    response, error = _get_with_retries(found["image"]["thumb_1024_url"])
    try:
        if response is None or response.status_code != 200:
            raise IOError(error or f"HTTP {response.status_code}")
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        result.update(status="download_error", error=str(e))
        return result

    from calculate_score import calculate_score  # loads the model on first use only

    fd, temp_filename = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        img.save(temp_filename)
        score, context = calculate_score(temp_filename, debug=False)
    finally:
        os.remove(temp_filename)

    result.update(score=score, context=context)
    if context == "Error":
        result.update(status="score_error", error="model could not open image")
        return result

    if save_image:
        folder_name = "test_images"
        os.makedirs(folder_name, exist_ok=True)
        # Turns "Waterfront 🌊" into "Waterfront" so file names stay safe.
        safe_context = context.split(" ")[0].replace("/", "_")
        final_filename = f"{folder_name}/{score:.2f}_{safe_context}_{lat:.4f}_{lon:.4f}.jpg"
        img.save(final_filename)
        print(f"  📸 Saved image to: {final_filename}")
    return result


def download_and_score(lat, lon, save_image=False):
    """Original interface used by build_graph.py: returns (score, context), with
    score None when no usable image was found."""
    result = fetch_and_score(lat, lon, save_image=save_image)
    status = result["status"]
    if status in ("found", "score_error"):  # score_error keeps the old (0.5, "Error")
        return result["score"], result["context"]
    if status == "api_error":
        print(f"API Error: {result['error']}")
        return None, "API Error"
    if status == "download_error":
        print(f"Error downloading/scoring image: {result['error']}")
        return None, "Download Error"
    return None, "No Data"


# --- TEST IT ---
if __name__ == "__main__":
    test_lat = 45.506
    test_lon = -73.539

    print(f"Searching for high-quality images since {MIN_DATE}...")

    # Set save_image=True here to test it out!
    score, context = download_and_score(test_lat, test_lon, save_image=True)

    print(f"Result -> Status: {context} | Score: {score}")
