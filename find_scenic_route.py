import osmnx as ox
import networkx as nx
import pandas as pd
import folium

# 1. LOAD THE GRAPH
filename = "scenic_graph_complete.graphml"
print(f"Loading {filename}...")
G = ox.load_graphml(filename)

# 2. DEFINE START & END POINTS BY NAME
# Just type the addresses or names of the places!
start_address = "McGill University, Montreal, Quebec"
end_address = "Place Jacques-Cartier, Montreal, Quebec"

# Exact coordinates can be used instead if you prefer:
# start_lat, start_lon = 45.5040, -73.5747
# end_lat, end_lon = 45.5050, -73.5530

print(f"Looking up coordinates for: {start_address}...")
# ox.geocode talks to the OpenStreetMap database to find the lat/lon
start_lat, start_lon = ox.geocode(start_address)

print(f"Looking up coordinates for: {end_address}...")
end_lat, end_lon = ox.geocode(end_address)

print(f"  -> S/'tart: {start_lat:.4f}, {start_lon:.4f}")
print(f"  -> End:   {end_lat:.4f}, {end_lon:.4f}")

print("Finding nearest street nodes to your coordinates...")
# Remember: ox expects Longitude (X) first, then Latitude (Y)
orig_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
dest_node = ox.distance.nearest_nodes(G, end_lon, end_lat)

# 3. PREPARE WEIGHTS 
for u, v, key, data in G.edges(keys=True, data=True):
    length = float(data.get("length", 1.0))
    score = float(data.get("scenic_score", 0.5)) 
    
    data["scenic_cost"] = length / max(score, 0.05)
    data["length"] = length 

# 4. CALCULATE BOTH ROUTES
print("Calculating routes...")
try:
    route_fast = nx.shortest_path(G, orig_node, dest_node, weight="length")
    route_scenic = nx.shortest_path(G, orig_node, dest_node, weight="scenic_cost")
except nx.NetworkXNoPath:
    print("❌ Error: No path could be found. Are both points inside your map area?")
    exit()

# 5. CONVERT TO GEOPANDAS
gdf_fast = ox.routing.route_to_gdf(G, route_fast)
gdf_scenic = ox.routing.route_to_gdf(G, route_scenic)

gdf_fast["scenic_score"] = pd.to_numeric(gdf_fast["scenic_score"], errors="coerce").fillna(0.5)
gdf_scenic["scenic_score"] = pd.to_numeric(gdf_scenic["scenic_score"], errors="coerce").fillna(0.5)

# 6. EXTRACT THE STATISTICS
fast_km = gdf_fast["length"].sum() / 1000
fast_score = gdf_fast["scenic_score"].mean()

scenic_km = gdf_scenic["length"].sum() / 1000
scenic_score = gdf_scenic["scenic_score"].mean()

print("\n--- 🏁 ROUTE RESULTS ---")
print(f"🔴 FASTEST ROUTE: {fast_km:.2f} km | Average Beauty: {fast_score:.2%} ({fast_score:.2f}/1.0)")
print(f"🔵 SCENIC ROUTE:  {scenic_km:.2f} km | Average Beauty: {scenic_score:.2%} ({scenic_score:.2f}/1.0)")
print("------------------------\n")

# 7. VISUALIZE BOTH ON ONE MAP
print("Generating comparison map...")
m = gdf_scenic.explore(
    color="#3388ff", 
    name="Scenic Route",
    tiles="CartoDB positron",
    style_kwds={"weight": 8, "opacity": 0.9}
)

gdf_fast.explore(
    m=m, 
    color="#ff3333",
    name="Fastest Route",
    style_kwds={"weight": 5, "opacity": 0.7, "dashArray": "5, 5"} 
)

folium.LayerControl().add_to(m)

output_file = "route_comparison.html"
m.save(output_file)
print(f"✅ Success! Open '{output_file}' to see both routes.")