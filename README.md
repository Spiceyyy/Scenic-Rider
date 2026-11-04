# 🚴 Scenic Viber (Scenic Router)
**Elevating urban navigation through Computer Vision and Graph Theory.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers/index)

## 🌟 Project Overview
Scenic Viber is a multi-objective routing engine designed for cyclists and pedestrians who prioritize "vibe" and aesthetics over the shortest path. While traditional GPS (Google Maps, Strava) focuses on distance or elevation, Scenic Viber analyzes the visual composition of streets to find the most "scenic" route.

The project currently focuses on the **Montreal Metropolitan Area**, analyzing over 430 km² of street-level data.

## 🧠 Technical Architecture

The system operates in three distinct phases:

### 1. Computer Vision Pipeline (The "Eyes")
We use a **Mask2Former** architecture to perform panoptic segmentation on street-level imagery. 
* **Backbone:** `Swin-Large` (Shifted Window Transformer). We chose the **Large** variant for its hierarchical attention mechanism, which excels at detecting fine-grained environmental features like tree canopies and water bodies.
* **Data Source:** Images are dynamically fetched via the **Mapillary API**.
* **Output:** Every image is converted into a feature vector representing the percentage of natural vs. urban elements.

### 2. Statistical Scenic Scoring (The "Brain")
Using a custom-weighted algorithm, we calculate a **Scenic Index ($S$)$** for every road segment:
$$S = \sum (w_{nature} \cdot P_{nature}) - (w_{urban} \cdot P_{urban})$$
Where $P$ represents the pixel percentage of specific classes (Trees, Sky, Water vs. Concrete, Fences, Cars).

### 3. Graph Optimization (The "Path")
* **NetworkX & OSMnx:** We model the city as a directed graph where nodes are intersections and edges are road segments.
* **Edge Weighting:** Instead of simple distance, edge weights are adjusted based on the Scenic Index.
* **Dijkstra's Variant:** The router finds the path that minimizes the "unscenic cost" while keeping the total distance within a user-defined threshold.

## 🛠 Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch, Hugging Face `transformers`
* **Geospatial:** OSMnx, GeoPandas, Shapely
* **API Integration:** Mapillary (Street-level imagery)
* **Data Analysis:** NumPy, Pandas

## 🚀 Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/scenic-viber.git](https://github.com/your-username/scenic-viber.git)
   cd scenic-viber
3. Install the required dependencies
   ```bash
   pip install -r requirements.txt
### Basic Usage:
```bash
from scenic_viber import RouteGenerator

# Initialize generator with Swin-Large backbone
rg = RouteGenerator(model="facebook/mask2former-swin-large-ade-panoptic")

# Define your start and end coordinates (Montreal)
start_coords = (45.5048, -73.5772) # McGill University
end_coords = (45.5071, -73.5875)   # Mount Royal Park

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
