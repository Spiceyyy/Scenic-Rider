from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import torch
import numpy as np
import urllib.request
import os

# --- 1. SETUP: Load Mask2Former (Mapillary Vistas) ---
# We use the 'swin-base' version. It is heavy (~400MB) but very smart.
# It knows 65 classes including "Bike Lane", "Curb", "Water", "Manhole".
print("Loading Mask2Former (Mapillary Vistas)...")
MODEL_NAME = "facebook/mask2former-swin-large-mapillary-vistas-semantic"

try:
    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_NAME)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME)
    model.eval()
except Exception as e:
    print(f"Error loading model. You might need to install dependencies: {e}")
    exit()

def get_id_by_name(keyword_list):
    """
    Helper to find Class IDs dynamically. 
    E.g., input ["water", "river"] -> finds the ID for "Water"
    """
    found_ids = []
    for id, label in model.config.id2label.items():
        for key in keyword_list:
            if key.lower() in label.lower():
                found_ids.append(id)
    return list(set(found_ids))

# PRE-CALCULATE CLASS GROUPS (Do this once to save time)
# Mapillary Vistas v1.2 Class Names
BIKE_LANE_IDS = get_id_by_name(["bike lane"])
WATER_IDS = get_id_by_name(["water"]) # Often Class 61
NATURE_IDS = get_id_by_name(["vegetation", "terrain", "mountain", "sand", "sky"])
ROAD_IDS = get_id_by_name(["road", "service lane", "crosswalk"]) # General asphalt
UGLY_IDS = get_id_by_name(["building", "wall", "fence", "car", "truck", "bus", "barrier"])

print(f"  - Bike Lane IDs: {BIKE_LANE_IDS}")
print(f"  - Water IDs: {WATER_IDS}")

def calculate_score(image_path, debug=False):
    """
    Returns Scenic Score (0.0-1.0) using Mapillary Vistas.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return 0.5, "Error"

    # --- INFERENCE (Specific to Mask2Former) ---
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Mask2Former outputs raw masks; we must post-process them into a map
        # target_sizes is needed to resize the output back to original image size
        predicted_semantic_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0] # Take the first (and only) item in batch
    
    pred_seg = predicted_semantic_map.numpy()

    # --- PIXEL COUNTING ---
    total = pred_seg.size
    
    bike_lane_px = sum(np.sum(pred_seg == i) for i in BIKE_LANE_IDS)
    water_px = sum(np.sum(pred_seg == i) for i in WATER_IDS)
    nature_px = sum(np.sum(pred_seg == i) for i in NATURE_IDS)
    road_px = sum(np.sum(pred_seg == i) for i in ROAD_IDS)
    ugly_px = sum(np.sum(pred_seg == i) for i in UGLY_IDS)

    # --- SCORING LOGIC ---
    
    # 1. Critical Bonus: Bike Lane
    # If there is an actual painted bike lane, that is huge for safety/enjoyment.
    bike_lane_ratio = bike_lane_px / total
    
    # 2. Blue Space
    water_ratio = water_px / total
    
    # 3. Green Space
    nature_ratio = nature_px / total

    # 4. Context Check
    status = "Neutral"
    bonus = 0
    penalty = 0

    if water_ratio > 0.10:
        status = "Waterfront 🌊"
        bonus = 0.3
        # Water forgives all ugliness
        ugly_px = ugly_px * 0.5 
        
    elif bike_lane_ratio > 0.05:
        # If >5% of the view is a bike lane, it's a dedicated path
        status = "Dedicated Bike Lane 🚲"
        bonus = 0.2
        # A bike lane next to a road is fine, so we reduce road penalty
        penalty = 0 
        
    elif nature_ratio > 0.40:
        status = "Green Tunnel 🌳"
        bonus = 0.1
    
    elif (road_px / total) > 0.40:
        status = "City St (No Bike Lane) 🚗"
        penalty = 0.2
    
    else:
        status = "Mixed / Urban"

    # Score Calc
    # Start neutral (0.5)
    # Add good stuff
    score = 0.5 + (nature_ratio * 0.5) + (water_ratio * 1.5) + (bike_lane_ratio * 2.0)
    
    # Subtract bad stuff
    score = score - (ugly_px / total * 0.5) + bonus - penalty
    
    # Clamp
    score = max(0.0, min(1.0, score))

    # --- DEBUG ---
    if debug:
        print(f"\n--- Vision Breakdown (Mapillary Vistas) ---")
        unique, counts = np.unique(pred_seg, return_counts=True)
        for class_id, count in zip(unique, counts):
            # Mask2Former ID mapping
            try:
                label = model.config.id2label[class_id]
            except:
                label = str(class_id)
                
            pct = count / total
            if pct > 0.02: # Show anything > 2%
                print(f"  - {label}: {pct:.1%}")
        print("------------------------------------------")

    return score, status

if __name__ == "__main__":
    # Auto-download a test image
    test_img = "ugly2.jpg"
    if not os.path.exists(test_img):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/images/deeplab1.png"
        urllib.request.urlretrieve(url, test_img)

    print(f"Testing Mask2Former on {test_img}...")
    s, c = calculate_score(test_img, debug=True)
    print(f"Status: {c} | Score: {s:.2%}")

# Print class mapping for reference
# print(model.config.id2label)