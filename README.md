# 🚴 Scenic Viber (Scenic Router)
**Elevating urban navigation through Computer Vision and Graph Theory.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers/index)

## 🌟 Project Overview
Scenic Viber is a multi-objective routing engine and interactive web app designed for cyclists and pedestrians who prioritize "vibe" and aesthetics over the absolute shortest path. While traditional GPS (Google Maps, Strava) focuses purely on distance or elevation, Scenic Viber analyzes the visual composition of streets using AI to find the most beautiful route.

The project currently maps the **Montreal Metropolitan Area**, automatically pulling and analyzing real-time street-level data to avoid ugly industrial roads and prioritize waterfronts, parks, and dedicated bike lanes.

## 🧠 Technical Architecture

The system operates in three distinct phases:

### 1. Computer Vision Pipeline (The "Eyes")
We use a **Mask2Former** architecture to perform semantic segmentation on street-level imagery. 
* **Model:** `facebook/mask2former-swin-large-mapillary-vistas-semantic` (65 classes including "Bike Lane", "Terrain", "Water", and "Wall").
* **Data Source:** High-quality panoramas dynamically fetched and filtered chronologically via the **Mapillary API**.
* **Output:** Every image is parsed to calculate the pixel percentages of nature, water, bike infrastructure, and urban clutter.

### 2. Statistical Scenic Scoring (The "Brain")
Images are scored on a scale of `0.0` to `1.0`. The baseline score of 0.5 is adjusted via a custom-weighted algorithm that heavily rewards blue/green spaces and penalizes concrete/cars.

To force the pathfinding algorithm to prefer beauty, we mathematically invert the relationship by calculating a custom edge weight (Cost). A high scenic score acts as a "discount" on the physical length of the road:

$$Cost = \frac{Length}{\max(Score, 0.05)}$$

### 3. Graph Optimization (The "Path")
* **NetworkX & OSMnx:** The city is modeled as a directed MultiDiGraph where nodes are intersections and edges are road segments.
* **Dijkstra's Algorithm:** By routing against our custom $Cost$ variable, the engine minimizes "unscenic cost," happily taking a slightly longer physical detour if it means riding along a beautiful canal or through a green tunnel.

## 🗂 Project Structure
The data pipeline is designed to be modular and fault-tolerant to handle API rate limits and long inference times.

* `build_graph.py`: The crawler. Downloads a neighborhood grid, fetches images, runs the AI model, and saves the data.
* `combine_graphs.py`: The stitcher. Merges individual neighborhood graphs into a master routable map.
* `app.py`: The frontend. A Streamlit dashboard that geocodes user addresses, calculates the Fastest vs. Scenic routes simultaneously, and visualizes them on an interactive Folium map.
* `/graphs`: Stores the generated `.graphml` data files.
* `/maps`: Stores standalone HTML visualizers.
* `/test_images`: An automated debug log containing the raw images the AI scored, overlaid with their classification.

## 🛠 Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch, Hugging Face `transformers`
* **Geospatial & Graph:** OSMnx, NetworkX, GeoPandas, Folium
* **Frontend UI:** Streamlit
* **API Integration:** Mapillary (Street-level imagery)

## 🚀 Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/scenic-viber.git](https://github.com/your-username/scenic-viber.git)
   cd scenic-viber
2. Instatll the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Add your Mapillary Developer Token to request.py (or a .env file).
### Usage Pipeline
1. Build your Map Data
Run the crawler to analyze a neighborhood (This may take a while depending on GPU hardware):
   ```bash
   python build_graph.py
(Optional: Run python combine_graphs.py if you have multiple neighborhood files).

# Generate the most scenic path
path = rg.get_best_vibe(start_coords, end_coords)
path.show()
```

## 🗄️ PostGIS ETL (Phase 1)
Loads the existing scored graphs (`graphs/*.graphml`) into PostgreSQL/PostGIS. No model inference is run; routing still reads GraphML.

```bash
pip install -r requirements-etl.txt
cp .env.example .env              # set POSTGRES_PASSWORD and DATABASE_URL
docker compose up -d --wait       # PostGIS on localhost:5433
python -m etl validate            # data-quality checks only, no DB
python -m etl init-db             # create schema "scenic" (idempotent)
python -m etl run                 # load master graph; re-runs are no-ops
python -m etl run --graph graphs/graph_le_plateau_mont_royal.graphml --strict
pytest                            # DB tests run only if TEST_DATABASE_URL is set
```

Tables: `data_source`, `score_method`, `road_node`, `road_segment`, `scenic_score`, `etl_run` (see `db/schema.sql`).
Re-runs upsert by natural key `(u, v, key)` / `osmid` and update a row only when its content hash changes.

### Database-first scoring
The database is the source of truth; the GraphML file for the app is generated from it.

```bash
python -m etl add-sector "Westmount, Quebec, Canada" --city Montreal   # OSM -> database, no file
python -m etl add-sector "Ville-Marie, Montreal, Canada" --link-only   # tag data already loaded
python -m etl sectors                  # every sector with its scoring progress
python -m etl score --dry-run          # how many streets still lack an image-based score
python -m etl score --limit 20 --sector westmount   # Mapillary + model, written to PostGIS
python -m etl export-graph [--city Montreal]        # graphs/db_export.graphml from the database
```

Rules: one search per street (both directions share it); only segments without a model score are
targeted; API/download errors never change a score and are retried next run; every lookup is logged
in `score_attempt` with its status, and scores keep their `image_id`, capture date and distance.

One database holds every city. A sector is one OSM place; `sector_segment` links segments to sectors, so
a street on a border is stored once and belongs to both. Sectors are downloaded with
`truncate_by_edge=True` so neighbouring sectors share border intersections and routes can cross them.
