"""Runtime configuration, read from environment variables (optionally a .env file)."""
import ast
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_SQL = PROJECT_ROOT / "db" / "schema.sql"
DEFAULT_GRAPH = PROJECT_ROOT / "graphs" / "master_scenic_graph.graphml"
TARGET_CRS = "EPSG:4326"

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def configure_osmnx():
    """Optional OVERPASS_URL in .env: use another Overpass server/mirror when the
    default public one (overpass-api.de) is busy or refuses automated requests."""
    url = os.environ.get("OVERPASS_URL")
    if url:
        import osmnx as ox
        ox.settings.overpass_url = url.rstrip("/")


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill it in, "
            "or export DATABASE_URL=postgresql://user:pass@host:port/db"
        )
    return url


def db_schema() -> str:
    schema = os.environ.get("SCENIC_DB_SCHEMA", "scenic")
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", schema):
        raise ValueError(f"Invalid SCENIC_DB_SCHEMA: {schema!r}")
    return schema


def scoring_model_name() -> str:
    """Read MODEL_NAME from calculate_score.py *without importing it*
    (importing would load the Swin-Large weights)."""
    source = (PROJECT_ROOT / "calculate_score.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MODEL_NAME" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return "unknown"
