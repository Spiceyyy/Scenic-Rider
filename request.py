import requests
import os
from PIL import Image
from io import BytesIO
from datetime import datetime
from calculate_score import calculate_score

# --- CONFIGURATION ---
MAPILLARY_TOKEN = "MLY|24871549075857260|49e4f5c2fe54fa7f7253ab756fa2239b"
SEARCH_URL = "https://graph.mapillary.com/images"
MIN_DATE = "2020-01-01"

def get_best_image_at_coordinate(lat, lon):
    delta = 0.0005 
    bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
    start_time = f"{MIN_DATE}T00:00:00Z"
    
    params = {
        "access_token": MAPILLARY_TOKEN,
        "fields": "id,thumb_1024_url,captured_at,is_pano", 
        "bbox": bbox,
        "start_captured_at": start_time,
        "limit": 10 
    }
    
    try:
        response = requests.get(SEARCH_URL, params=params)
        data = response.json()
        candidates = data.get("data", [])
        
        if not candidates:
            return None 
            
        def score_candidate(img):
            pano_bonus = 1 if img.get("is_pano") else 0 
            date_str = img.get("captured_at", "2000-01-01")
            return (pano_bonus, date_str)
            
        best_image = max(candidates, key=score_candidate)
        return best_image["thumb_1024_url"]
        
    except Exception as e:
        print(f"API Error: {e}")
        return None

# --- NEW: Added save_image parameter ---
def download_and_score(lat, lon, save_image=False):
    image_url = get_best_image_at_coordinate(lat, lon)
    
    if image_url:
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Temporary save for the AI model
            temp_filename = "temp_scan.jpg"
            img.save(temp_filename)
            
            # Get the score
            score, context = calculate_score(temp_filename, debug=False)
            
            # --- THE NEW SAVING LOGIC ---
            if save_image:
                # 1. Create a folder so your main directory doesn't get messy
                folder_name = "test_images"
                os.makedirs(folder_name, exist_ok=True)
                
                # 2. Clean the context string (e.g., remove emojis or spaces so files don't break)
                # This turns "Waterfront 穴" into "Waterfront"
                safe_context = context.split(" ")[0].replace("/", "_")
                
                # 3. Create a beautiful filename
                # Example: test_images/0.85_Waterfront_45.5060_-73.5390.jpg
                final_filename = f"{folder_name}/{score:.2f}_{safe_context}_{lat:.4f}_{lon:.4f}.jpg"
                img.save(final_filename)
                print(f"  📸 Saved image to: {final_filename}")

            # Clean up the temp file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

            return score, context
            
        except Exception as e:
            print(f"Error downloading/scoring image: {e}")
            return None, "Download Error"
    else:
        return None, "No Data"

# --- TEST IT ---
if __name__ == "__main__":
    test_lat = 45.506
    test_lon = -73.539
    
    print(f"Searching for high-quality images since {MIN_DATE}...")
    
    # Set save_image=True here to test it out!
    score, context = download_and_score(test_lat, test_lon, save_image=True)
    
    print(f"Result -> Status: {context} | Score: {score}")