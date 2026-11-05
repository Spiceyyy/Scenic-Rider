import osmnx as ox
import matplotlib.pyplot as plt

# 1. SETUP: Where do you want to look?
# Using your previous coordinates (Laurier Park area)
lat, lon = 45.460838, -73.605154 # Canal Lachine
dist = 500 # Radius in meters

print(f"Downloading bike network for {dist}m radius...")

# 2. DOWNLOAD THE RAW GRAPH
# network_type="bike" gets cycleways, paths, and bike-friendly streets
G = ox.graph_from_point((lat, lon), dist=dist, network_type="bike")

print(f"Stats: {len(G.nodes)} intersections, {len(G.edges)} street segments.")

# 3. VISUALIZE IT (The Skeleton)
# node_size=0 -> Hide the dots (intersections)
# edge_linewidth=1.5 -> Make streets visible
# edge_color="#FFFFFF" -> White lines
# bgcolor="#111111" -> Dark background
fig, ax = ox.plot_graph(
    G, 
    node_size=0, 
    edge_color="#3399ff", # Nice bright blue
    edge_linewidth=1.5,
    bgcolor="#111111",
    show=True # Opens a window with the map
)