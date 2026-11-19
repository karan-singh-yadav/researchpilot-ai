import io
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from sqlalchemy import select

from app.models import DocumentChunk, UploadedDocument
from app.rag.embeddings import EmbeddingError, HashingEmbedder
from app.rag.index import SqlIndex
from app.rag.service import DocumentError, DocumentService, sanitize_filename
from tests.pdf_factory import ORCHESTRA, make_pdf
from tests.rag_env import make_rag


@pytest.fixture()
def rag(tmp_path):
    r = make_rag(tmp_path)
    yield r
    r.env.engine.dispose()


def row(rag, doc_id) -> UploadedDocument:
    db = rag.env.SF()
    try:
        return db.get(UploadedDocument, doc_id)
    finally:
        db.close()


def chunk_count(rag, doc_id) -> int:
    db = rag.env.SF()
    try:
        return db.query(DocumentChunk).filter_by(document_id=doc_id).count()
    finally:
        db.close()


def test_upload_processes_into_searchable_chunks_with_embeddings(rag):
    doc = rag.upload("Orchestra X.pdf", ORCHESTRA, title="Orchestra-X Whitepaper")
    assert doc["status"] == "ready" and doc["page_count"] == 3 and doc["chunk_count"] >= 3
    assert doc["embedding_model"] == "hash-512-v1" and doc["doc_metadata"] == {
        "title": "Orchestra-X Whitepaper"
    }
    db = rag.env.SF()
    chunks = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == doc["id"])
        .order_by(DocumentChunk.chunk_index)
    ).all()
    assert {c.page for c in chunks} == {1, 2, 3} and all(
        len(c.embedding) == 512 * 4 for c in chunks
    )
    assert [c.id for c in chunks] == [f"{doc['id']}:{i}" for i in range(len(chunks))]
    db.close()


@pytest.mark.parametrize(
    "name,data,code,status",
    [
        ("a.pdf", b"", "empty_file", 400),
        ("a.txt", make_pdf(["hello there"]), "invalid_file_type", 415),
        ("a.pdf", b"This is plain text, not a PDF.", "not_pdf", 415),
        ("a.pdf", b"PK\x03\x04" + b"0" * 100, "not_pdf", 415),
    ],
)
def test_upload_validation(rag, name, data, code, status):
    with pytest.raises(DocumentError) as ei:
        rag.service.upload(name, data)
    assert (ei.value.code, ei.value.status) == (code, status)
    db = rag.env.SF()
    assert db.query(UploadedDocument).count() == 0  # nothing was stored
    db.close()


def test_size_limit_is_enforced_before_any_processing(tmp_path):
    rag = make_rag(tmp_path, max_upload_mb=1)
    big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)
    with pytest.raises(DocumentError) as ei:
        rag.service.upload("big.pdf", big)
    assert (ei.value.code, ei.value.status) == ("file_too_large", 413)
    assert not Path(rag.settings.upload_dir).exists() or not list(
        Path(rag.settings.upload_dir).iterdir()
    )
    rag.env.engine.dispose()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\me\\paper.pdf", "paper.pdf"),
        ("evil\x00name\x1b[31m.pdf", "evilname[31m.pdf"),
        ("", "document.pdf"),
        ("...", "document.pdf"),
        ("a" * 400 + ".pdf", "a" * 196 + ".pdf"),
    ],
)
def test_filenames_are_sanitised(raw, expected):
    assert sanitize_filename(raw) == expected


def test_files_are_stored_under_a_generated_name_inside_the_upload_dir_only(rag):
    doc, _ = rag.service.upload("../../../tmp/evil.pdf", make_pdf(ORCHESTRA))
    stored = list(Path(rag.settings.upload_dir).iterdir())
    assert [p.name for p in stored] == [f"{doc['id']}.pdf"]
    assert doc["filename"] == "evil.pdf" and not Path("/tmp/evil.pdf").exists()
    assert not list(Path(rag.settings.upload_dir).glob("*.part"))  # no temp files left behind


def test_identical_bytes_are_detected_as_duplicates(rag):
    pdf = make_pdf(ORCHESTRA)
    first, dup1 = rag.service.upload("one.pdf", pdf)
    second, dup2 = rag.service.upload("renamed-copy.pdf", pdf)
    assert (
        (dup1, dup2) == (False, True)
        and second["id"] == first["id"]
        and second["filename"] == "one.pdf"
    )
    db = rag.env.SF()
    assert db.query(UploadedDocument).count() == 1
    db.close()
    other, dup3 = rag.service.upload(
        "two.pdf", make_pdf(["Completely different content for another file."])
    )
    assert dup3 is False and other["id"] != first["id"]


def test_failed_documents_report_safe_errors_and_leave_no_chunks(rag):
    scanned = rag.upload("scan.pdf", ["", ""])
    assert (
        scanned["status"] == "failed"
        and "OCR" in scanned["error"]
        and chunk_count(rag, scanned["id"]) == 0
    )
    broken, _ = rag.service.upload("broken.pdf", b"%PDF-1.4\n" + b"\x00junk" * 80)
    assert (
        broken["status"] == "failed"
        and "Traceback" not in broken["error"]
        and chunk_count(rag, broken["id"]) == 0
    )
    w = PdfWriter()
    for pg in PdfReader(io.BytesIO(make_pdf(ORCHESTRA))).pages:
        w.add_page(pg)
    w.encrypt(user_password="pw", owner_password="own")
    buf = io.BytesIO()
    w.write(buf)
    locked, _ = rag.service.upload("locked.pdf", buf.getvalue())
    assert locked["status"] == "failed" and "password" in locked["error"]


def test_a_failed_upload_can_be_retried_with_the_same_bytes(rag):
    class Flaky(HashingEmbedder):
        fail = True

        def embed(self, texts):
            if Flaky.fail:
                raise EmbeddingError("provider returned HTTP 503", retryable=True)
            return super().embed(texts)

    emb = Flaky(512)
    svc = DocumentService(
        rag.env.SF, rag.settings, emb, SqlIndex(rag.env.SF, emb.name), executor="inline"
    )
    pdf = make_pdf(ORCHESTRA)
    doc, _ = svc.upload("a.pdf", pdf)
    assert (
        doc["status"] == "failed"
        and "Embedding failed" in doc["error"]
        and "reindex" in doc["error"]
    )
    Flaky.fail = False
    again, dup = svc.upload(
        "a.pdf", pdf
    )  # same bytes, previous attempt failed => reprocess same record
    assert dup is False and again["id"] == doc["id"] and again["status"] == "ready"


def test_reindex_recovers_a_failed_embedding_and_switches_embedding_model(rag):
    doc = rag.upload("a.pdf", ORCHESTRA)

    class Down:
        name = "hash-512-v1"

        def embed(self, texts):
            raise EmbeddingError("provider down", retryable=True)

    rag.service.embedder = Down()
    out = rag.service.reindex(doc["id"])
    assert out["status"] == "failed" and "provider down" in out["error"]
    assert chunk_count(rag, doc["id"]) > 0  # chunk text survives so a retry needs no PDF re-parse
    rag.service.embedder = HashingEmbedder(256)
    rag.service.index = SqlIndex(rag.env.SF, "hash-256-v1")
    fixed = rag.service.reindex(doc["id"])
    assert (
        fixed["status"] == "ready"
        and fixed["embedding_model"] == "hash-256-v1"
        and fixed["error"] is None
    )
    db = rag.env.SF()
    c = db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == doc["id"])).first()
    assert len(c.embedding) == 256 * 4
    db.close()


def test_reindex_guards(rag):
    with pytest.raises(DocumentError) as ei:
        rag.service.reindex("nope")
    assert ei.value.status == 404
    doc = rag.upload("a.pdf", ORCHESTRA)
    db = rag.env.SF()
    db.get(UploadedDocument, doc["id"]).status = "processing"
    db.commit()
    db.close()
    with pytest.raises(DocumentError) as ei:
        rag.service.reindex(doc["id"])
    assert ei.value.status == 409


def test_delete_removes_row_chunks_and_file_and_is_safe_against_late_processing(rag):
    doc = rag.upload("a.pdf", ORCHESTRA)
    path = Path(rag.settings.upload_dir) / f"{doc['id']}.pdf"
    assert path.exists() and chunk_count(rag, doc["id"]) > 0
    assert rag.service.delete(doc["id"]) is True
    assert not path.exists() and row(rag, doc["id"]) is None and chunk_count(rag, doc["id"]) == 0
    assert rag.service.delete(doc["id"]) is False
    rag.service.process(
        doc["id"]
    )  # a worker finishing after deletion must not crash or resurrect it
    assert row(rag, doc["id"]) is None and chunk_count(rag, doc["id"]) == 0


def test_missing_stored_file_is_reported_not_raised(rag):
    doc = rag.upload("a.pdf", ORCHESTRA)
    (Path(rag.settings.upload_dir) / f"{doc['id']}.pdf").unlink()
    rag.service.process(doc["id"])
    assert (
        row(rag, doc["id"]).status == "failed" and "could not be read" in row(rag, doc["id"]).error
    )


def test_background_processing_returns_immediately_then_becomes_ready(tmp_path):
    r = make_rag(tmp_path, executor="thread")
    doc, _ = r.service.upload("a.pdf", make_pdf(ORCHESTRA))
    assert doc["status"] in ("processing", "ready")
    r.service.wait(doc["id"])
    assert row(r, doc["id"]).status == "ready"
    r.service.shutdown()
    r.env.engine.dispose()
