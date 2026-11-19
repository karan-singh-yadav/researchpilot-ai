"""Stable ID assignment: sources get S1.., evidence gets E1.. (never reused, never invented)."""

from app.schemas.evidence import EvidenceItem, SourceRecord
from app.tools.urls import url_key


class SourceRegistry:
    def __init__(self) -> None:
        self._by_key: dict[str, SourceRecord] = {}
        self._by_ref: dict[str, SourceRecord] = {}
        self._evidence: list[EvidenceItem] = []

    def has_url(self, url: str) -> bool:
        return url_key(url) in self._by_key

    def keys(self) -> frozenset[str]:
        return frozenset(self._by_key)

    def add_source(self, **fields) -> SourceRecord:
        key = url_key(fields["url"])
        if key in self._by_key:
            return self._by_key[key]
        rec = SourceRecord(ref=f"S{len(self._by_ref) + 1}", **fields)
        self._by_key[key] = rec
        self._by_ref[rec.ref] = rec
        return rec

    def add_evidence(self, source_ref: str, **fields) -> EvidenceItem:
        if source_ref not in self._by_ref:
            raise KeyError(f"unknown source {source_ref}")
        item = EvidenceItem(id=f"E{len(self._evidence) + 1}", source_ref=source_ref, **fields)
        self._evidence.append(item)
        return item

    @property
    def sources(self) -> dict[str, SourceRecord]:
        return dict(self._by_ref)

    @property
    def evidence(self) -> list[EvidenceItem]:
        return list(self._evidence)

    def evidence_map(self) -> dict[str, EvidenceItem]:
        return {e.id: e for e in self._evidence}

    @classmethod
    def restore(cls, sources: dict[str, dict], evidence: list[dict]) -> "SourceRegistry":
        """Rebuild from checkpointed state, preserving every S#/E# id exactly."""
        reg = cls()
        for ref, raw in sources.items():
            rec = SourceRecord(**raw)
            reg._by_key[url_key(rec.url)] = rec
            reg._by_ref[ref] = rec
        reg._evidence = [EvidenceItem(**e) for e in evidence]
        return reg
