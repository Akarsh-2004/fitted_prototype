import sqlite3
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional
from pipeline.config import settings
from pipeline.database.models import (
    CREATE_ITEMS_TABLE_SQL,
    CREATE_JOBS_TABLE_SQL,
    MIGRATION_COLUMNS,
)

@contextmanager
def get_db_connection():
    """Context manager for SQLite database connection."""
    db_path = settings.sqlite_db_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _apply_migrations(cursor: sqlite3.Cursor) -> None:
    """Idempotently add any columns introduced after the initial schema."""
    for table, column, ddl in MIGRATION_COLUMNS:
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        if column not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db():
    """Initializes the database schema if tables do not exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(CREATE_ITEMS_TABLE_SQL)
        cursor.execute(CREATE_JOBS_TABLE_SQL)
        _apply_migrations(cursor)
        conn.commit()

# --- Job Operations ---

def create_job(job_id: str, status: str, scene_type: Optional[str] = None, original_image_path: Optional[str] = None) -> None:
    """Creates a new background processing job record."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO jobs (job_id, status, scene_type, original_image_path)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, status, scene_type, original_image_path)
        )
        conn.commit()

def update_job(
    job_id: str,
    status: str,
    scene_type: Optional[str] = None,
    detected_items: Optional[List[Dict[str, Any]]] = None,
    result: Optional[List[Dict[str, Any]]] = None,
    error: Optional[str] = None
) -> None:
    """Updates status, scene_type, results, or error for an existing job."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build dynamic update query
        fields = ["status = ?"]
        params = [status]
        
        if scene_type is not None:
            fields.append("scene_type = ?")
            params.append(scene_type)
        if detected_items is not None:
            fields.append("detected_items = ?")
            params.append(json.dumps(detected_items))
        if result is not None:
            fields.append("result = ?")
            params.append(json.dumps(result))
        if error is not None:
            fields.append("error = ?")
            params.append(error)
            
        params.append(job_id)
        query = f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?"
        
        cursor.execute(query, tuple(params))
        conn.commit()

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a background processing job by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        data = dict(row)
        if data.get("detected_items"):
            data["detected_items"] = json.loads(data["detected_items"])
        if data.get("result"):
            data["result"] = json.loads(data["result"])
        return data

# --- Wardrobe Item Operations ---

def insert_wardrobe_item(item_id: str, item_data: Dict[str, Any]) -> None:
    """Inserts a fully processed wardrobe item into the database and, when
    possible, generates the 1024x1024 Outfit Composer alignment PNG."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO wardrobe_items (
                id, garment_type, fit, material, construction, pattern, subtype, brand,
                style, occasion, season, archetype, layering_role, pairing_suggestions,
                colors, tags, image_path, scene_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                item_data.get("garment_type", "unknown"),
                item_data.get("fit"),
                item_data.get("material"),
                item_data.get("construction"),
                item_data.get("pattern"),
                item_data.get("subtype"),
                item_data.get("brand"),
                item_data.get("style"),
                item_data.get("occasion"),
                item_data.get("season"),
                item_data.get("archetype"),
                item_data.get("layering_role"),
                json.dumps(item_data.get("pairing_suggestions", [])),
                json.dumps(item_data.get("colors", [])),
                json.dumps(item_data.get("tags", [])),
                item_data.get("image_path"),
                item_data.get("scene_type")
            )
        )
        conn.commit()

    # Best-effort composer alignment - never fail ingest if Pillow choke.
    try:
        from pipeline.composer.alignment import align_item, aligned_relative_path

        composable = dict(item_data)
        composable["id"] = item_id
        result = align_item(composable)
        if result is not None:
            _, category = result
            update_wardrobe_item_alignment(item_id, aligned_relative_path(item_id), category)
    except Exception as exc:  # noqa: BLE001 - alignment is a non-fatal post-process
        print(f"[composer] alignment skipped for {item_id}: {exc}")


def update_wardrobe_item_alignment(item_id: str, aligned_image_path: str, composer_category: str) -> None:
    """Persists the aligned PNG path + composer category for a wardrobe item."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE wardrobe_items SET aligned_image_path = ?, composer_category = ? WHERE id = ?",
            (aligned_image_path, composer_category, item_id),
        )
        conn.commit()

def get_wardrobe_item(item_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a wardrobe item by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM wardrobe_items WHERE id = ?", (item_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        data = dict(row)
        data["pairing_suggestions"] = json.loads(data["pairing_suggestions"]) if data.get("pairing_suggestions") else []
        data["colors"] = json.loads(data["colors"]) if data.get("colors") else []
        data["tags"] = json.loads(data["tags"]) if data.get("tags") else []
        return data

def get_all_jobs() -> List[Dict[str, Any]]:
    """Retrieves all background processing jobs in reverse chronological order."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50")
        rows = cursor.fetchall()
        jobs = []
        for r in rows:
            data = dict(r)
            if data.get("detected_items"):
                data["detected_items"] = json.loads(data["detected_items"])
            if data.get("result"):
                data["result"] = json.loads(data["result"])
            jobs.append(data)
        return jobs

def get_all_wardrobe_items() -> List[Dict[str, Any]]:
    """Retrieves all ingested wardrobe items."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM wardrobe_items ORDER BY created_at DESC")
        rows = cursor.fetchall()
        items = []
        for r in rows:
            data = dict(r)
            data["pairing_suggestions"] = json.loads(data["pairing_suggestions"]) if data.get("pairing_suggestions") else []
            data["colors"] = json.loads(data["colors"]) if data.get("colors") else []
            data["tags"] = json.loads(data["tags"]) if data.get("tags") else []
            items.append(data)
        return items

def delete_wardrobe_item(item_id: str) -> bool:
    """Deletes a wardrobe item by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM wardrobe_items WHERE id = ?", (item_id,))
        conn.commit()
        return cursor.rowcount > 0
