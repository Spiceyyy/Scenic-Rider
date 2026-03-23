import osmnx as ox
import networkx as nx
import time
import re
import os
from request import download_and_score 

# 1. SETUP: Define the Area
PLACE_NAME = "Le Sud-Ouest, Montreal, Canada"

GRAPH_FOLDER = "graphs"
os.makedirs(GRAPH_FOLDER, exist_ok=True) # Creates the folder safely

# This turns "Le Plateau-Mont-Royal, Montreal, Canada" into "le_plateau_mont_royal"
safe_name = PLACE_NAME.split(',')[0].strip().lower()
safe_name = re.sub(r'[^a-z0-9]', '_', safe_name)
safe_name = re.sub(r'_+', '_', safe_name) # Remove double underscores

backup_filename = os.path.join(GRAPH_FOLDER, f"backup_{safe_name}.graphml")
final_filename = os.path.join(GRAPH_FOLDER, f"graph_{safe_name}.graphml")

print(f"Downloading bike network for: {PLACE_NAME}")
G = ox.graph_from_place(PLACE_NAME, network_type="bike")
print(f"Graph loaded. Analyzing {len(G.edges)} street segments...")

print(f"Downloading bike network for: {PLACE_NAME}")
G = ox.graph_from_place(PLACE_NAME, network_type="bike")
print(f"Graph loaded. Analyzing {len(G.edges)} street segments...")

# We convert to a list so we can modify the graph while iterating
edges = list(G.edges(data=True))
total_edges = len(edges)

for i, (u, v, data) in enumerate(edges):
    
    # Check if we already scored this
    if "scenic_score" in data:
        continue

    print(f"[{i+1}/{total_edges}] Processing edge {u}->{v}...", end=" ")

    # --- NEW STRATEGY: Check 3 points along the road ---
    # Instead of just one midpoint, we try 25%, 50%, and 75%.
    check_points = [0.5, 0.25, 0.75] 
    
    found_score = None
    found_context = "No Image"
    
    # We default geometry to None in case it's a straight line
    geom = data.get("geometry", None)

    for ratio in check_points:
        # 1. Calculate Coordinate
        if geom:
            # If curved road, interpolate along the curve
            pt = geom.interpolate(ratio, normalized=True)
            lat, lon = pt.y, pt.x
        else:
            # If straight line (no geometry), we can only check the middle
            # (Math for 25% on a straight line is possible but rarely needed for short segments)
            node_u = G.nodes[u]
            node_v = G.nodes[v]
            lat = (node_u['y'] + node_v['y']) / 2
            lon = (node_u['x'] + node_v['x']) / 2
            
            # If we only have start/end, checking 3 times is redundant, so break after first
            if ratio != 0.5: 
                break 

        # 2. Try to Download
        try:
            # Check this specific coordinate
            score, context = download_and_score(lat, lon, save_image=False)
            
            # If we found something, SAVE IT and STOP looking!
            if score is not None:
                found_score = score
                found_context = context
                print(f"  -> Found at {ratio:.0%}: {score:.2f} ({context})")
                break  # <--- Critical: Stop the loop once we have an image
        
        except Exception as e:
            print(f"Error at {ratio}: {e}")

    # 3. FINAL DECISION
    if found_score is None:
        # If we checked 3 times and STILL found nothing:
        # Fallback to trusting the map tags (Inference)
        highway_type = data.get("highway", "")
        if highway_type in ["cycleway", "path", "footway"]:
             found_score = 0.8  # Assume paths are nice
             found_context = "Inferred (Path)"
        else:
             found_score = 0.5  # Neutral for empty streets
             found_context = "No Data"
        print(f"  -> {found_context}")
    
    # 4. Save to Graph
    G[u][v][0]["scenic_score"] = found_score
    G[u][v][0]["scenic_context"] = found_context
    
    # Rate Limit
    time.sleep(0.5)

    # Periodic Save
    if i % 10 == 0:
        ox.save_graphml(G, backup_filename)
    

# 3. FINAL SAVE
print("Analysis Complete! Saving Graph...")
ox.save_graphml(G, final_filename)
print(f"✅ Saved beautifully to '{final_filename}'")