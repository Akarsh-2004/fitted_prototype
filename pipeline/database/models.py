import json
from datetime import datetime

# Direct SQL commands to create our database schema
CREATE_ITEMS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS wardrobe_items (
    id TEXT PRIMARY KEY,
    garment_type TEXT NOT NULL,
    fit TEXT,
    material TEXT,
    construction TEXT,
    pattern TEXT,
    subtype TEXT,
    brand TEXT,
    style TEXT,
    occasion TEXT,
    season TEXT,
    archetype TEXT,
    layering_role TEXT,
    pairing_suggestions TEXT, -- JSON string
    colors TEXT,              -- JSON string (RGB + weight list)
    tags TEXT,                -- JSON string of list
    image_path TEXT,          -- local crop PNG path
    aligned_image_path TEXT,  -- 1024x1024 outfit-composer PNG
    composer_category TEXT,   -- hat | top | bottom | shoes
    scene_type TEXT,          -- flat_single, flat_multi, single_person, group_photo
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Columns added after the initial schema, applied as idempotent ALTER TABLEs.
MIGRATION_COLUMNS = [
    ("wardrobe_items", "aligned_image_path", "TEXT"),
    ("wardrobe_items", "composer_category", "TEXT"),
]

CREATE_JOBS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,      -- queued, processing, requires_confirmation, requires_selection, completed, failed
    scene_type TEXT,           -- flat_single, flat_multi, single_person, group_photo
    original_image_path TEXT,
    detected_items TEXT,       -- JSON string (for Scenario 2 confirmation crops)
    result TEXT,               -- JSON string (final ingested items list)
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
