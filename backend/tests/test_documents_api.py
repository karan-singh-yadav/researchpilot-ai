import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_documents, get_llm, get_retriever
from app.config import Settings, get_settings
from app.database import get_db
from app.main import create_app
from app.models import ResearchSession
from app.security.rate_limit import limiter
from app.services.llm.fake import FakeLLM
from tests.pdf_factory import ORCHESTRA, make_pdf, wrap
from tests.rag_env import COOKING, make_rag


@pytest.fixture()
def api(tmp_path):
    rag = make_rag(tmp_path, max_upload_mb=1)
    app = create_app()

    def _db():
        db = rag.env.SF()
        try:
            yield db
            db.commit()
        finally:
            db.close()

    app.dependency_overrides.update(
        {
            get_db: _db,
            get_documents: lambda: rag.service,
            get_retriever: lambda: rag.retriever,
            get_llm: lambda: FakeLLM(),
            get_settings: lambda: rag.settings,
        }
    )
    limiter.reset()
    c = TestClient(app, raise_server_exceptions=False)
    yield type("A", (), {"c": c, "rag": rag, "app": app})
    rag.env.engine.dispose()
    get_settings.cache_clear()


def upload(c, name="orchestra.pdf", pages=ORCHESTRA):
    return c.post("/api/documents", files={"file": (name, make_pdf(pages), "application/pdf")})


def test_upload_list_get_and_page_filtered_chunks(api):
    r = upload(api.c)
    assert r.status_code == 202, r.text
    doc = r.json()["document"]
    assert r.json()["duplicate"] is False and doc["status"] == "ready" and doc["page_count"] == 3
    assert "storage" not in str(doc).lower() and ".pdf" not in doc["id"]  # no server paths leak
    upload(api.c, "cooking.pdf", COOKING)
    lst = api.c.get("/api/documents").json()
    assert lst["total"] == 2 and lst["items"][0]["filename"] == "cooking.pdf"  # newest first
    assert api.c.get("/api/documents", params={"q": "orch"}).json()["total"] == 1
    assert api.c.get("/api/documents", params={"status": "failed"}).json()["total"] == 0
    assert api.c.get("/api/documents", params={"status": "bogus"}).status_code == 422
    assert api.c.get("/api/documents", params={"limit": 1}).json()["items"].__len__() == 1
    assert api.c.get(f"/api/documents/{doc['id']}").json()["filename"] == "orchestra.pdf"
    chunks = api.c.get(f"/api/documents/{doc['id']}/chunks", params={"page": 2}).json()
    assert chunks and {c["page"] for c in chunks} == {2} and "87%" in chunks[0]["text"]
    assert api.c.get("/api/documents/nope").status_code == 404
    assert api.c.get("/api/documents/nope/chunks").status_code == 404


def test_duplicate_upload_returns_200_with_the_existing_document(api):
    first = upload(api.c).json()["document"]
    r = upload(api.c, "copy-of-it.pdf")
    assert (
        r.status_code == 200
        and r.json()["duplicate"] is True
        and r.json()["document"]["id"] == first["id"]
    )
    assert api.c.get("/api/documents").json()["total"] == 1


def test_upload_errors_use_the_error_envelope(api):
    def post(name, data):
        return api.c.post("/api/documents", files={"file": (name, data, "application/pdf")})

    for name, data, code, status in [
        ("a.txt", make_pdf(["x y z"]), "invalid_file_type", 415),
        ("a.pdf", b"not a pdf at all", "not_pdf", 415),
        ("a.pdf", b"", "empty_file", 400),
        ("a.pdf", b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 5), "file_too_large", 413),
    ]:
        r = post(name, data)
        assert r.status_code == status and r.json()["error"]["code"] == code, (name, r.text[:200])
    assert api.c.post("/api/documents").status_code == 422  # missing file field
    assert api.c.get("/api/documents").json()["total"] == 0


def test_failed_processing_is_visible_on_the_document(api):
    r = upload(api.c, "scan.pdf", ["", ""])
    assert r.status_code == 202
    d = r.json()["document"]
    assert d["status"] == "failed" and "OCR" in d["error"]
    assert api.c.get("/api/documents", params={"status": "failed"}).json()["total"] == 1


def test_search_returns_scored_chunks_with_document_and_page(api):
    upload(api.c)
    upload(api.c, "cooking.pdf", COOKING)
    r = api.c.post(
        "/api/documents/search", json={"query": "benchmark completed multi-step tasks", "top_k": 3}
    )
    assert r.status_code == 200
    top = r.json()["chunks"][0]
    assert (
        top["filename"] == "orchestra.pdf"
        and top["page"] == 2
        and top["score"] > 0.1
        and "87%" in top["text"]
    )
    only = api.c.post(
        "/api/documents/search", json={"query": "agents graph", "document_ids": ["nope"]}
    ).json()
    assert only["chunks"] == []
    for bad in (
        {"query": "ab"},
        {"query": "valid query", "top_k": 21},
        {"query": "valid query", "page_min": 0},
    ):
        assert api.c.post("/api/documents/search", json=bad).status_code == 422


def test_ask_answers_with_verified_page_citations(api):
    doc = upload(api.c).json()["document"]
    r = api.c.post(
        "/api/documents/ask",
        json={
            "question": "What share of tasks did the supervised graph complete in the benchmark?",
            "document_ids": [doc["id"]],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answered"] is True and "87%" in body["answer"] and body["llm"].startswith("fake:")
    cit = body["citations"][0]
    assert (
        cit["filename"] == "orchestra.pdf" and cit["page"] == 2 and cit["document_id"] == doc["id"]
    )
    chunk = next(c for c in body["retrieved"] if c["chunk_id"] == cit["chunk_id"])
    assert " ".join(cit["quote"].split()) in " ".join(chunk["text"].split())  # quote is verbatim


def test_ask_handles_unanswerable_and_empty_cases_without_inventing(api):
    upload(api.c)
    r = api.c.post(
        "/api/documents/ask", json={"question": "zebra giraffe savannah migration patterns?"}
    ).json()
    assert r["answered"] is False and r["citations"] == []
    none = api.c.post(
        "/api/documents/ask", json={"question": "anything at all here", "document_ids": []}
    ).json()
    assert none["answered"] is False and none["retrieved"] == []
    assert api.c.post("/api/documents/ask", json={"question": "ab"}).status_code == 422


def test_delete_and_reindex(api):
    doc = upload(api.c).json()["document"]
    r = api.c.post(f"/api/documents/{doc['id']}/reindex")
    assert r.status_code == 202 and r.json()["status"] == "ready"
    assert api.c.post("/api/documents/nope/reindex").status_code == 404
    assert api.c.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert api.c.get(f"/api/documents/{doc['id']}").status_code == 404
    assert api.c.delete(f"/api/documents/{doc['id']}").status_code == 404
    assert (
        api.c.post("/api/documents/search", json={"query": "orchestra graph agents"}).json()[
            "chunks"
        ]
        == []
    )


def test_a_document_in_use_by_a_running_session_cannot_be_deleted(api):
    doc = upload(api.c).json()["document"]
    db = api.rag.env.SF()
    db.add(
        ResearchSession(
            query="long running research about documents",
            mode="document",
            status="running",
            document_ids=[doc["id"]],
        )
    )
    db.commit()
    db.close()
    r = api.c.delete(f"/api/documents/{doc['id']}")
    assert r.status_code == 409 and r.json()["error"]["code"] == "document_in_use"
    assert api.c.get(f"/api/documents/{doc['id']}").status_code == 200


def test_endpoints_require_the_api_key_and_report_missing_configuration(api, monkeypatch):
    app2 = create_app()  # no documents service attached => documented 503, not a crash
    c2 = TestClient(app2, raise_server_exceptions=False)
    r = c2.post("/api/documents/search", json={"query": "anything here"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "documents_not_configured"
    keyed = Settings(_env_file=None, api_keys="k1")
    api.app.dependency_overrides[get_settings] = lambda: keyed
    monkeypatch.setenv("API_KEYS", "k1")
    get_settings.cache_clear()
    api.app.dependency_overrides.pop(get_settings, None)
    for method, path in [
        ("get", "/api/documents"),
        ("post", "/api/documents/search"),
        ("post", "/api/documents/ask"),
        ("delete", "/api/documents/x"),
    ]:
        assert getattr(api.c, method)(path).status_code == 401, path
    assert api.c.get("/api/documents", headers={"X-API-Key": "k1"}).status_code == 200


def test_long_filenames_and_page_counts_do_not_break_the_response_model(api):
    long_name = ("report-" * 40) + ".pdf"
    r = api.c.post(
        "/api/documents",
        files={"file": (long_name, make_pdf([wrap("lorem ipsum " * 50)] * 4), "application/pdf")},
    )
    assert (
        r.status_code == 202
        and len(r.json()["document"]["filename"]) <= 200
        and r.json()["document"]["page_count"] == 4
    )
