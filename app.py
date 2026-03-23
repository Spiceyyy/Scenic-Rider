import streamlit as st
import osmnx as ox
import networkx as nx
import pandas as pd
import folium
from streamlit_folium import st_folium

# 1. PAGE SETUP
# This configures the browser tab and layout
st.set_page_config(page_title="Scenic Route Finder", page_icon="🚲", layout="centered")
st.title("🚲 Scenic Route Finder")
st.markdown("Type in two locations to find the most beautiful bike route between them!")

# 2. CACHE THE GRAPH (CRITICAL STEP)
# @st.cache_resource tells Streamlit to only run this function ONCE.
# Otherwise, it would take 10 seconds to reload the GraphML file every time you click a button!
@st.cache_resource
def load_and_prep_graph():
    # G = ox.load_graphml("scenic_graph_complete.graphml")
    G = ox.load_graphml("graphs/master_scenic_graph.graphml")
    
    # Pre-calculate the scenic costs so we don't have to do it during routing
    for u, v, key, data in G.edges(keys=True, data=True):
        length = float(data.get("length", 1.0))
        score = float(data.get("scenic_score", 0.5)) 
        data["scenic_cost"] = length / max(score, 0.05)
        data["length"] = length 
        
    return G

# Load the graph invisibly in the background
G = load_and_prep_graph()

# 3. CREATE THE USER INTERFACE
# Put the text boxes side-by-side using columns
col1, col2 = st.columns(2)
start_address = col1.text_input("Start Address", "McGill University, Montreal")
end_address = col2.text_input("End Address", "Old Port, Montreal")

# --- NEW: Initialize our "Backpack" (Session State) ---
if "route_calculated" not in st.session_state:
    st.session_state.route_calculated = False

# 4. THE ACTION BUTTON
if st.button("Generate Route", type="primary"):
    with st.spinner("Finding coordinates and calculating the best routes..."):
        try:
            # --- GEOCODING ---
            start_lat, start_lon = ox.geocode(start_address)
            end_lat, end_lon = ox.geocode(end_address)

            orig_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
            dest_node = ox.distance.nearest_nodes(G, end_lon, end_lat)

            # --- ROUTING ---
            route_fast = nx.shortest_path(G, orig_node, dest_node, weight="length")
            route_scenic = nx.shortest_path(G, orig_node, dest_node, weight="scenic_cost")

            gdf_fast = ox.routing.route_to_gdf(G, route_fast)
            gdf_scenic = ox.routing.route_to_gdf(G, route_scenic)

            gdf_fast["scenic_score"] = pd.to_numeric(gdf_fast["scenic_score"], errors="coerce").fillna(0.5)
            gdf_scenic["scenic_score"] = pd.to_numeric(gdf_scenic["scenic_score"], errors="coerce").fillna(0.5)

            # --- SAVE DATA TO MEMORY (Not the map!) ---
            # Save the raw shapes so we can redraw them safely
            st.session_state.gdf_fast = gdf_fast
            st.session_state.gdf_scenic = gdf_scenic
            
            # Save the stats
            st.session_state.fast_km = gdf_fast["length"].sum() / 1000
            st.session_state.fast_score = gdf_fast["scenic_score"].mean()
            st.session_state.scenic_km = gdf_scenic["length"].sum() / 1000
            st.session_state.scenic_score = gdf_scenic["scenic_score"].mean()
            
            # Tell the app we have data ready to show
            st.session_state.route_calculated = True 

        except Exception as e:
            st.error(f"⚠️ Could not find a route. Error details: {e}")

# 5. DISPLAY THE RESULTS (Outside the button click!)
if st.session_state.route_calculated:
    st.divider() 
    
    # Display Stats
    stat_col1, stat_col2 = st.columns(2)
    stat_col1.metric(
        "🔴 Fastest Route", 
        f"{st.session_state.fast_km:.2f} km", 
        f"Beauty Score: {st.session_state.fast_score:.0%}"
    )
    stat_col2.metric(
        "🔵 Scenic Route", 
        f"{st.session_state.scenic_km:.2f} km", 
        f"Beauty Score: {st.session_state.scenic_score:.0%}"
    )

    # --- REBUILD THE MAP HERE ---
    m = st.session_state.gdf_scenic.explore(
        color="#3388ff", 
        name="Scenic Route",
        tiles="CartoDB positron",
        style_kwds={"weight": 8, "opacity": 0.9}
    )

    st.session_state.gdf_fast.explore(
        m=m, 
        color="#ff3333",
        name="Fastest Route",
        style_kwds={"weight": 5, "opacity": 0.7, "dashArray": "5, 5"} 
    )

    folium.LayerControl().add_to(m)

    # --- THE MAGIC FIX: returned_objects=[] ---
    # This tells the map to stay quiet and not trigger a page reload!
    st_folium(m, width=700, height=500, returned_objects=[])