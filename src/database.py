from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "benchmark_results.sqlite3"


def _normalize_path(db_path: str | Path | None = None) -> Path:
    return Path(db_path) if db_path is not None else DATABASE_PATH


def init_db(db_path: str | Path | None = None) -> Path:
    database_path = _normalize_path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                category TEXT,
                original_prompt TEXT NOT NULL,
                compressed_prompt TEXT,
                original_tokens INTEGER,
                compressed_tokens INTEGER,
                compression_ratio REAL,
                compression_latency_seconds REAL,
                baseline_latency_seconds REAL,
                optimized_latency_seconds REAL,
                latency_saved_ms REAL,
                reference_response TEXT,
                candidate_response TEXT,
                bert_f1_score REAL,
                carbon_footprint_g REAL,
                simulated_cost REAL,
                token_savings INTEGER,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.commit()

    return database_path


def log_result(db_path: str | Path | None = None, **telemetry: Any) -> int:
    database_path = init_db(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    payload_json = json.dumps(telemetry, ensure_ascii=True, default=str)

    columns = [
        "created_at",
        "category",
        "original_prompt",
        "compressed_prompt",
        "original_tokens",
        "compressed_tokens",
        "compression_ratio",
        "compression_latency_seconds",
        "baseline_latency_seconds",
        "optimized_latency_seconds",
        "latency_saved_ms",
        "reference_response",
        "candidate_response",
        "bert_f1_score",
        "carbon_footprint_g",
        "simulated_cost",
        "token_savings",
        "payload_json",
    ]
    values = [created_at, *(telemetry.get(column) for column in columns[1:-1]), payload_json]

    placeholders = ", ".join("?" for _ in columns)
    column_list = ", ".join(columns)

    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            f"INSERT INTO benchmark_results ({column_list}) VALUES ({placeholders})",
            values,
        )
        connection.commit()
        return int(cursor.lastrowid)