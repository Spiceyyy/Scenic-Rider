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
```
app.py                 Streamlit app: address A -> B, fastest vs scenic route on a map
request.py             Mapillary search + download (token from MAPILLARY_TOKEN), calls the model
calculate_score.py     Mask2Former segmentation -> scenic score 0..1
etl/                   Database pipeline: python -m etl <command> (see below)
db/schema.sql          PostGIS tables + the v_segment_score view used for routing
tests/                 pytest suite (DB tests use a throwaway schema)
docker-compose.yml     Local PostGIS database
graphs/                GraphML files: db_export.graphml (generated, used by the app) and the
                       original per-neighbourhood graphs loaded into the database
maps/                  HTML maps written by the scripts below
samples/               Example street photos for trying the model
build_graph.py, combine_graph.py   Original file-based pipeline (superseded by etl/, kept for reference)
find_scenic_route.py, visualize_map.py, preview_network.py   Standalone map/route scripts
```

## 🛠 Tech Stack
* **Language:** Python, SQL
* **Data:** PostgreSQL + PostGIS (Docker), psycopg2, pytest
* **Deep Learning:** PyTorch, Hugging Face `transformers`
* **Geospatial & Graph:** OSMnx, NetworkX, GeoPandas, Shapely, Folium
* **Frontend UI:** Streamlit
* **API Integration:** Mapillary (street-level imagery), OpenStreetMap / Overpass

## 🚀 Getting Started
```bash
git clone https://github.com/Spiceyyy/Scenic-Rider.git
cd Scenic-Rider
pip install -r requirements.txt -r requirements-etl.txt
cp .env.example .env          # add MAPILLARY_TOKEN, POSTGRES_PASSWORD, DATABASE_URL
streamlit run app.py          # graphs/db_export.graphml, else the committed master graph
```
`db_export.graphml` is generated from the database (`python -m etl export-graph`, next section).

## 🗄️ PostGIS ETL
Loads the existing scored graphs (`graphs/*.graphml`) into PostgreSQL/PostGIS. Loading never runs model inference; scoring new streets is a separate command (`etl score`).

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

Tables: `data_source`, `score_method`, `road_node`, `road_segment`, `scenic_score`, `etl_run`, `score_attempt`, `sector`, `sector_segment`, `green_area`, plus the `v_segment_score` view (see `db/schema.sql`).
Re-runs upsert by natural key `(u, v, key)` / `osmid` and update a row only when its content hash changes.

### Database-first scoring
The database is the source of truth; the GraphML file for the app is generated from it.

```bash
python -m etl add-sector "Westmount, Quebec, Canada" --city Montreal   # OSM -> database, no file
python -m etl add-sector "Ville-Marie, Montreal, Canada" --link-only   # tag data already loaded
python -m etl sectors                  # every sector with its scoring progress
python -m etl green                    # parks/woods from OSM (park bonus, see below)
python -m etl score --dry-run          # how many streets still lack an image-based score
python -m etl score --limit 20 --sector westmount   # Mapillary + model, written to PostGIS
python -m etl score --rescore-legacy --bbox=MINLON,MINLAT,MAXLON,MAXLAT  # redo old scores with no recorded photo
python -m etl export-graph [--city Montreal]        # graphs/db_export.graphml from the database
```

Rules: one search per street (both directions share it); only segments without a model score are
targeted; API/download errors never change a score and are retried next run; every lookup is logged
in `score_attempt` with its status, and scores keep their `image_id`, capture date and distance.

One database holds every city. A sector is one OSM place; `sector_segment` links segments to sectors, so
a street on a border is stored once and belongs to both. Sectors are downloaded with
`truncate_by_edge=True` so neighbouring sectors share border intersections and routes can cross them.

**Final score (`v_segment_score` view).** Raw scores are never modified; routing uses a derived score:
- `green_share` = fraction of the segment inside a park/wood (`green_area`, from OSM)
- the "Inferred (Path)" 0.8 guess only counts for segments mostly (>= 50%) inside green; bike lanes
  beside streets fall back to 0.5
- `final_score = min(1, base_score + 0.15 * green_share)`; `export-graph` writes it as `scenic_score`
  (`scenic_raw_score` keeps the original). If overpass-api.de refuses requests, set `OVERPASS_URL` in `.env`.
