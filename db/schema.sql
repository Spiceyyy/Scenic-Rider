-- Scenic Viber storage schema (PostgreSQL + PostGIS).
-- Idempotent: safe to run repeatedly. Table names are unqualified; the ETL
-- sets search_path to the target schema (default "scenic") before running it.
-- All geometries are EPSG:4326 (WGS84 lon/lat), matching the OSMnx graphs.

CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;

-- One row per distinct source file (identified by content hash).
CREATE TABLE IF NOT EXISTS data_source (
    source_id        SERIAL PRIMARY KEY,
    file_name        TEXT        NOT NULL,
    file_sha256      CHAR(64)    NOT NULL UNIQUE,
    created_with     TEXT,                 -- e.g. "OSMnx 2.0.7"
    graph_created_at TEXT,                 -- as recorded in the GraphML header
    source_crs       TEXT        NOT NULL,
    node_count       INTEGER     NOT NULL,
    edge_count       INTEGER     NOT NULL,
    registered_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- How a score was produced (vision model, OSM-tag inference, neutral default...).
CREATE TABLE IF NOT EXISTS score_method (
    method_id   SERIAL PRIMARY KEY,
    method_code TEXT NOT NULL,
    model_name  TEXT NOT NULL DEFAULT '',  -- HF model id for method_code = 'model'
    description TEXT NOT NULL,
    UNIQUE (method_code, model_name)
);

CREATE TABLE IF NOT EXISTS road_node (
    osmid        BIGINT PRIMARY KEY,
    street_count INTEGER,
    highway      TEXT,
    geom         geometry(Point, 4326) NOT NULL,
    source_id    INTEGER NOT NULL REFERENCES data_source (source_id),
    row_hash     CHAR(32) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS road_node_geom_gix ON road_node USING GIST (geom);

-- Directed road segment = one edge (u, v, key) of the OSMnx MultiDiGraph.
CREATE TABLE IF NOT EXISTS road_segment (
    u           BIGINT  NOT NULL REFERENCES road_node (osmid),
    v           BIGINT  NOT NULL REFERENCES road_node (osmid),
    key         INTEGER NOT NULL,
    osm_way_ids BIGINT[],
    name        TEXT,
    highway     TEXT,                      -- ';'-joined when OSMnx merged several ways
    oneway      BOOLEAN,
    reversed    TEXT,
    length_m    DOUBLE PRECISION NOT NULL CHECK (length_m >= 0),
    osm_tags    JSONB NOT NULL DEFAULT '{}'::jsonb,  -- lanes, maxspeed, bridge, ...
    geom        geometry(LineString, 4326) NOT NULL,
    source_id   INTEGER NOT NULL REFERENCES data_source (source_id),
    row_hash    CHAR(32) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (u, v, key)
);
CREATE INDEX IF NOT EXISTS road_segment_geom_gix ON road_segment USING GIST (geom);
CREATE INDEX IF NOT EXISTS road_segment_highway_idx ON road_segment (highway);

-- Current scenic score per segment (scores are taken from existing graph files;
-- this pipeline never runs model inference).
CREATE TABLE IF NOT EXISTS scenic_score (
    u             BIGINT  NOT NULL,
    v             BIGINT  NOT NULL,
    key           INTEGER NOT NULL,
    score         DOUBLE PRECISION NOT NULL CHECK (score >= 0 AND score <= 1),
    context       TEXT,                    -- raw label from calculate_score / build_graph
    context_label TEXT,                    -- same label without emoji, for BI tools
    method_id     INTEGER NOT NULL REFERENCES score_method (method_id),
    source_id     INTEGER NOT NULL REFERENCES data_source (source_id),
    row_hash      CHAR(32) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (u, v, key),
    FOREIGN KEY (u, v, key) REFERENCES road_segment (u, v, key) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS scenic_score_method_idx ON scenic_score (method_id);

CREATE TABLE IF NOT EXISTS etl_run (
    run_id      SERIAL PRIMARY KEY,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status      TEXT NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    source_file TEXT NOT NULL,
    source_id   INTEGER REFERENCES data_source (source_id),
    stats       JSONB,                     -- validation issues + insert/update counts
    error       TEXT
);
