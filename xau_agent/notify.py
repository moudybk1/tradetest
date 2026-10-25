"""Optional Telegram alerts. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable."""

from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self, token: str | None, chat_id: str | None):
        self.token = token
        self.chat_id = chat_id

    @classmethod
    def from_env(cls) -> "Notifier":
        return cls(os.getenv("TELEGRAM_BOT_TOKEN") or None, os.getenv("TELEGRAM_CHAT_ID") or None)

    def send(self, text: str) -> None:
        log.info("NOTIFY: %s", text)
        if not (self.token and self.chat_id):
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text[:4000]},
                timeout=10,
            )
        except requests.RequestException as exc:
            log.warning("telegram send failed: %s", exc)
