"""High-impact economic events (CPI, NFP, FOMC ...) from the free ForexFactory weekly feed."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


@dataclass(frozen=True)
class Event:
    when: datetime  # UTC
    currency: str
    impact: str
    title: str

    def label(self) -> str:
        return f"{self.when:%a %H:%M} UTC {self.currency} {self.title}"


def parse_events(raw: list[dict], currencies: tuple[str, ...], impacts: tuple[str, ...]) -> list[Event]:
    events = []
    for item in raw:
        if item.get("country") not in currencies or item.get("impact") not in impacts:
            continue
        try:
            when = datetime.fromisoformat(item["date"]).astimezone(timezone.utc)
        except (KeyError, ValueError):
            continue
        events.append(Event(when, item["country"], item["impact"], item.get("title", "?")))
    return sorted(events, key=lambda e: e.when)


class EconomicCalendar:
    def __init__(
        self,
        cache_file: Path,
        currencies: tuple[str, ...] = ("USD",),
        impacts: tuple[str, ...] = ("High",),
        ttl_seconds: int = 3600,
    ):
        self.cache_file = cache_file
        self.currencies = currencies
        self.impacts = impacts
        self.ttl = ttl_seconds

    def _load_raw(self) -> list[dict] | None:
        fresh = self.cache_file.exists() and time.time() - self.cache_file.stat().st_mtime < self.ttl
        if not fresh:
            try:
                resp = requests.get(FEED_URL, timeout=15, headers={"User-Agent": "xau-agent/1.0"})
                resp.raise_for_status()
                data = resp.json()
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                self.cache_file.write_text(json.dumps(data))
                return data
            except (requests.RequestException, ValueError) as exc:
                log.warning("calendar fetch failed: %s", exc)
        if self.cache_file.exists():
            # A stale cache is still better than nothing, as long as it covers this week.
            try:
                return json.loads(self.cache_file.read_text())
            except ValueError:
                return None
        return None

    def events(self) -> list[Event] | None:
        raw = self._load_raw()
        if raw is None:
            return None
        return parse_events(raw, self.currencies, self.impacts)

    def blackout(self, now: datetime, before_min: int, after_min: int) -> tuple[bool | None, list[Event]]:
        """(True, events) inside a blackout, (False, []) clear, (None, []) calendar unavailable."""
        events = self.events()
        if events is None:
            return None, []
        hits = [
            e
            for e in events
            if e.when - timedelta(minutes=before_min) <= now <= e.when + timedelta(minutes=after_min)
        ]
        return bool(hits), hits

    def upcoming(self, now: datetime, hours: int = 24) -> list[Event]:
        events = self.events() or []
        return [e for e in events if now - timedelta(hours=1) <= e.when <= now + timedelta(hours=hours)]
