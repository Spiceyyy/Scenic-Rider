import osmnx as ox

# Define your target area (Start small!)
place_name = "Le Plateau-Mont-Royal, Montreal, Canada"

# Download the graph
# network_type='bike' filters for paths that allow bikes
G = ox.graph_from_place(place_name, network_type='bike')

# Project it to meters (important for measuring distance later)
G = ox.project_graph(G)

# Plot it to verify
ox.plot_graph(G)