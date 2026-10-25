"""SQLite journal: every signal, agent decision, order and block is recorded here."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


class Journal:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS day_equity (
                day TEXT PRIMARY KEY,
                start_equity REAL NOT NULL
            );
            """
        )

    def log(self, kind: str, **data: Any) -> None:
        self.db.execute(
            "INSERT INTO events (ts, kind, data) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), kind, json.dumps(data, default=str)),
        )
        self.db.commit()

    def day_start_equity(self, day: date, equity: float) -> float:
        """Return the equity recorded at the first check of `day`, recording it if new."""
        self.db.execute(
            "INSERT OR IGNORE INTO day_equity (day, start_equity) VALUES (?, ?)", (day.isoformat(), equity)
        )
        self.db.commit()
        row = self.db.execute("SELECT start_equity FROM day_equity WHERE day = ?", (day.isoformat(),)).fetchone()
        return float(row[0])

    def count_today(self, kind: str, day: date) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM events WHERE kind = ? AND substr(ts, 1, 10) = ?", (kind, day.isoformat())
        ).fetchone()
        return int(row[0])

    def recent(self, limit: int = 20) -> list[tuple[str, str, dict]]:
        rows = self.db.execute("SELECT ts, kind, data FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [(ts, kind, json.loads(data)) for ts, kind, data in rows]
