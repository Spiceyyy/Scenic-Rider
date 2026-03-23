import osmnx as ox
import pandas as pd
import geopandas as gpd
import os

# 1. LOAD THE MASTER GRAPH
filename = "master_scenic_graph.graphml"
print(f"Loading {filename}... (This might take a moment depending on how big it is!)")
try:
    G = ox.load_graphml(filename)
except FileNotFoundError:
    print(f"❌ Error: Could not find '{filename}'. Did you run combine_graphs.py first?")
    exit()

print(f"Graph loaded! Nodes: {len(G.nodes)} | Edges (Streets): {len(G.edges)}")

# 2. CONVERT TO DATAFRAME
print("Converting graph to map shapes...")
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

# 3. PREPARE THE DATA
# Ensure the scenic score is a number. 
# We fill missing values with 0.5 (Neutral) just in case a street slipped through.
gdf_edges["scenic_score"] = pd.to_numeric(gdf_edges["scenic_score"], errors='coerce').fillna(0.5)

print("Painting the map...")

# 4. CREATE THE INTERACTIVE MAP
# We use slightly thinner lines (weight: 3) because a master graph is visually dense.
m = gdf_edges.explore(
    column="scenic_score",
    cmap="RdYlGn",               # Red (Ugly) -> Yellow (Neutral) -> Green (Beautiful)
    tiles="CartoDB positron",   
    style_kwds={"weight": 3, "opacity": 0.8}  
)

MAPS_FOLDER = "maps"
os.makedirs(MAPS_FOLDER, exist_ok=True)

# 5. SAVE TO HTML
output_file = os.path.join(MAPS_FOLDER, "master_map_view.html")
m.save(output_file)
print(f"✅ Success! Your giant map is ready. Open '{output_file}' in your web browser.")