import osmnx as ox
import networkx as nx
import glob
import os

GRAPH_FOLDER = "graphs"

print(f"Looking for graph files in the '{GRAPH_FOLDER}' folder...")
# This looks for: graphs/graph_*.graphml
search_pattern = os.path.join(GRAPH_FOLDER, "graph_*.graphml")
file_list = glob.glob(search_pattern)

if not file_list:
    print("❌ No graph files found! Run build_graph.py first.")
    exit()

print(f"Found {len(file_list)} files:")
for f in file_list:
    print(f" - {f}")

# 1. Load all graphs into a list
graphs = []
for file in file_list:
    print(f"Loading {file} into memory...")
    graphs.append(ox.load_graphml(file))

# 2. Stitch them together
print("Stitching graphs together... (This might take a moment)")
# nx.compose_all merges overlapping nodes and combines all edges
master_graph = nx.compose_all(graphs)

# 3. Save the Master Graph
master_filename = os.path.join(GRAPH_FOLDER, "master_scenic_graph.graphml")
print(f"Saving combined master graph to {master_filename}...")
ox.save_graphml(master_graph, master_filename)

print(f"✅ Success! Your combined map has {len(master_graph.nodes)} nodes and {len(master_graph.edges)} edges.")