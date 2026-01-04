import requests
import os
from PIL import Image
from io import BytesIO
from calculate_score import calculate_score

# --- CONFIGURATION ---
MAPILLARY_TOKEN = "MLY|24871549075857260|49e4f5c2fe54fa7f7253ab756fa2239b"
SEARCH_URL = "https://graph.mapillary.com/images"

def get_image_at_coordinate(lat, lon):
    """
    1. Define a tiny box around the coordinate.
    2. Ask Mapillary for images inside that box.
    3. Return the URL of the first image found.
    """
    # Create a small bounding box (approx 20-30 meters)
    delta = 0.001
    bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
    
    params = {
        "access_token": MAPILLARY_TOKEN,
        "fields": "id,thumb_1024_url", # Get the ID and a medium-sized thumbnail
        "bbox": bbox,
        "limit": 1 # We only need one sample
    }
    
    try:
        response = requests.get(SEARCH_URL, params=params)
        data = response.json()
        
        if "data" in data and len(data["data"]) > 0:
            # Return the URL of the found image
            return data["data"][0]["thumb_1024_url"]
        else:
            return None # No image found here
    except Exception as e:
        print(f"API Error: {e}")
        return None

def download_and_score(lat, lon):
    # 1. Get URL
    image_url = get_image_at_coordinate(lat, lon)
    
    if image_url:
        print(f"Found image at {lat}, {lon}")
        
        # 2. Download Image into Memory (don't save to disk to save space)
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # 3. Save temporarily for your scorer (since your function expects a path)
        temp_filename = "temp_scan.jpg"
        img.save(temp_filename)
        
        # 4. Run YOUR Model
        # (Assuming your previous function is named calculate_score)
        score, context = calculate_score(temp_filename, debug=True)
        # filename = f"Score_{score:.2f}__Lat_{lat}_Lon_{lon}.jpg"
        # img.save(filename)

        # if os.remove(temp_filename):
        #     os.remove(temp_filename)

        return score, context
    else:
        print(f"No street view data for {lat}, {lon}")
        return None, "No Data"
        

# --- TEST IT ---
# Coordinates for a spot in Montreal (e.g., Canal Lachine area)
test_lat = 45.506
test_lon = -73.539


score, context = download_and_score(test_lat, test_lon)
print(f"Mapillary Score: {score}")