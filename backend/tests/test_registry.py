import pytest

from app.services.registry import SourceRegistry
from tests.helpers import NOW


def add(reg, url, **kw):
    return reg.add_source(title="T", url=url, retrieved_at=NOW, **kw)


def test_sequential_stable_ids_and_dedupe_by_url_key():
    reg = SourceRegistry()
    a = add(reg, "https://www.a.test/x/?utm_source=n#f")
    b = add(reg, "https://b.test/y")
    again = add(reg, "http://a.test/x")  # same page, different decoration
    assert (a.ref, b.ref) == ("S1", "S2") and again is a
    assert (
        len(reg.sources) == 2
        and reg.has_url("https://a.test/x")
        and not reg.has_url("https://c.test")
    )


def test_evidence_ids_and_unknown_source_rejected():
    reg = SourceRegistry()
    s = add(reg, "https://a.test/x")
    e1 = reg.add_evidence(s.ref, passage="p" * 30, claim="c")
    e2 = reg.add_evidence(s.ref, passage="q" * 30, claim="c")
    assert (e1.id, e2.id) == ("E1", "E2") and set(reg.evidence_map()) == {"E1", "E2"}
    with pytest.raises(KeyError):
        reg.add_evidence("S99", passage="x" * 30, claim="c")
    assert reg.keys() == frozenset({"a.test/x"})
