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

-- ---------------------------------------------------------------------------
-- Database-first scoring (python -m etl score). Added after the initial schema,
-- hence ALTER ... IF NOT EXISTS so existing databases upgrade in place.
-- ---------------------------------------------------------------------------
ALTER TABLE etl_run ADD COLUMN IF NOT EXISTS job TEXT NOT NULL DEFAULT 'graph_load';

-- Scores written by the live scorer have no source file, but do have an image.
ALTER TABLE scenic_score ALTER COLUMN source_id DROP NOT NULL;
ALTER TABLE scenic_score ADD COLUMN IF NOT EXISTS image_id TEXT;
ALTER TABLE scenic_score ADD COLUMN IF NOT EXISTS image_captured_at TIMESTAMPTZ;
ALTER TABLE scenic_score ADD COLUMN IF NOT EXISTS image_distance_m DOUBLE PRECISION;
ALTER TABLE scenic_score ADD COLUMN IF NOT EXISTS scored_run_id INTEGER REFERENCES etl_run (run_id);

-- Every Mapillary lookup, successful or not. One attempt covers both directions
-- of a street (all segments sharing the same geometry).
CREATE TABLE IF NOT EXISTS score_attempt (
    attempt_id       BIGSERIAL PRIMARY KEY,
    run_id           INTEGER NOT NULL REFERENCES etl_run (run_id),
    street_hash      CHAR(32) NOT NULL,     -- md5 of direction-independent geometry
    attempted_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    status           TEXT NOT NULL CHECK (status IN
                       ('found', 'no_images', 'api_error', 'download_error', 'score_error')),
    checkpoint       DOUBLE PRECISION,      -- position along the street that was searched
    http_status      INTEGER,
    image_id         TEXT,
    error            TEXT,
    segments_updated INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS score_attempt_street_idx ON score_attempt (street_hash, attempted_at);

-- ---------------------------------------------------------------------------
-- Sectors (python -m etl add-sector). One database for every city; a sector is
-- one OSM place (e.g. a borough). A segment on a border can belong to several
-- sectors, hence the link table instead of a column on road_segment.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sector (
    sector_id         SERIAL PRIMARY KEY,
    name              TEXT NOT NULL UNIQUE,   -- OSM place query, e.g. "Le Sud-Ouest, Montreal, Canada"
    slug              TEXT NOT NULL UNIQUE,   -- short id used on the CLI, e.g. "le_sud_ouest"
    city              TEXT NOT NULL,
    network_type      TEXT NOT NULL DEFAULT 'bike',
    boundary          geometry(MultiPolygon, 4326),
    osm_downloaded_at TIMESTAMPTZ,            -- NULL when only linked to existing data
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sector_boundary_gix ON sector USING GIST (boundary);
CREATE INDEX IF NOT EXISTS sector_city_idx ON sector (city);

CREATE TABLE IF NOT EXISTS sector_segment (
    sector_id INTEGER NOT NULL REFERENCES sector (sector_id) ON DELETE CASCADE,
    u         BIGINT  NOT NULL,
    v         BIGINT  NOT NULL,
    key       INTEGER NOT NULL,
    PRIMARY KEY (sector_id, u, v, key),
    FOREIGN KEY (u, v, key) REFERENCES road_segment (u, v, key) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS sector_segment_edge_idx ON sector_segment (u, v, key);

-- ---------------------------------------------------------------------------
-- Green areas (python -m etl green): parks, gardens, woods from OpenStreetMap.
-- Used to tell the router "this segment runs through a park", which a single
-- street photo cannot.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS green_area (
    green_id  BIGSERIAL PRIMARY KEY,
    osm_type  TEXT   NOT NULL,              -- way / relation
    osm_id    BIGINT NOT NULL,
    name      TEXT,
    kind      TEXT   NOT NULL,              -- e.g. leisure=park, natural=wood
    geom      geometry(MultiPolygon, 4326) NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (osm_type, osm_id)
);
CREATE INDEX IF NOT EXISTS green_area_geom_gix ON green_area USING GIST (geom);

-- ---------------------------------------------------------------------------
-- Final score used for routing (raw scores in scenic_score are never modified;
-- the rules live here so they are visible, versioned and easy to change):
--   green_share  = fraction of the segment's length inside a green area
--   base_score   = raw score, except the "Inferred (Path)" 0.8 guess only counts
--                  when the segment is mostly (>= 50%) inside a green area;
--                  bike lanes beside streets fall back to the neutral 0.5
--   final_score  = min(1, base_score + 0.15 * green_share)
--                  (unscored segments use 0.5 as base only if they touch green)
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_segment_score;
CREATE VIEW v_segment_score AS
WITH g AS (
    SELECT r.u, r.v, r.key,
           -- COALESCE first: LEAST() ignores NULLs, so LEAST(1, NULL) would be 1
           LEAST(1.0, COALESCE(gs.inside_m / NULLIF(ST_Length(r.geom::geography), 0), 0))
               AS green_share
    FROM road_segment r
    LEFT JOIN LATERAL (
        SELECT ST_Length(ST_Intersection(r.geom, ST_Union(ga.geom))::geography) AS inside_m
        FROM green_area ga
        WHERE ST_Intersects(r.geom, ga.geom)
    ) gs ON TRUE
), b AS (
    SELECT g.u, g.v, g.key, g.green_share, s.score AS raw_score, s.context, m.method_code,
           CASE WHEN s.score IS NULL THEN NULL
                WHEN m.method_code = 'inferred_tag' AND g.green_share < 0.5 THEN 0.5
                ELSE s.score END AS base_score
    FROM g
    LEFT JOIN scenic_score s USING (u, v, key)
    LEFT JOIN score_method m USING (method_id)
)
SELECT u, v, key, raw_score, base_score, green_share, context, method_code,
       CASE WHEN base_score IS NULL AND green_share = 0 THEN NULL
            ELSE LEAST(1.0, COALESCE(base_score, 0.5) + 0.15 * green_share) END AS final_score
FROM b;
