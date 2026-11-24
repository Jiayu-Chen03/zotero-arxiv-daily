"""Persistent arXiv delivery history for the daily workflow."""

from datetime import datetime
import hmac
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


def arxiv_id(value: str) -> str:
    """Normalize RSS IDs and Atom links to the version-independent arXiv ID."""
    value = value.removeprefix("oai:arXiv.org:")
    if "://" in value:
        value = urlsplit(value).path.removeprefix("/abs/")
    return re.sub(r"v\d+$", "", value)


class DigestState:
    def __init__(self, path: Path, key: str, recipient: str):
        self.path = path
        self.key = key.encode()
        self.recipient = recipient.strip().lower()
        self.sent = set()
        self.last_retrieved_at = None
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.sent = set(data["sent"])
            if data["last_retrieved_at"]:
                self.last_retrieved_at = datetime.fromisoformat(data["last_retrieved_at"])

    def _fingerprint(self, paper_id: str) -> str:
        # The state branch is public. Scope the fingerprints to this recipient.
        message = f"arxiv-digest\0{self.recipient}\0{paper_id}".encode()
        return hmac.digest(self.key, message, "sha256").hex()

    def contains(self, paper_id: str) -> bool:
        return self._fingerprint(paper_id) in self.sent

    def record(self, paper_ids: list[str], retrieved_at: datetime | None) -> None:
        if not paper_ids and retrieved_at is None:
            return
        self.sent.update(map(self._fingerprint, paper_ids))
        if retrieved_at is not None:
            self.last_retrieved_at = retrieved_at
        data = {
            "sent": sorted(self.sent),
            "last_retrieved_at": self.last_retrieved_at.isoformat() if self.last_retrieved_at else None,
        }
        pending = self.path.with_suffix(".tmp")
        pending.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        pending.replace(self.path)
