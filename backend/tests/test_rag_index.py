import numpy as np
import pytest
from sqlalchemy import select

from app.models import DocumentChunk, UploadedDocument
from app.rag.embeddings import EmbeddingError, HashingEmbedder
from app.rag.index import ChromaIndex, SqlIndex
from app.rag.retriever import LexicalReranker, RetrievalError, Retriever
from app.rag.types import DocumentChunk as Chunk
from tests.rag_env import COOKING, FINANCE, ORCHESTRA, make_rag


@pytest.fixture()
def rag(tmp_path):
    r = make_rag(tmp_path)
    r.docs = {
        "orchestra": r.upload("orchestra.pdf", ORCHESTRA),
        "cooking": r.upload("cooking.pdf", COOKING),
        "finance": r.upload("finance.pdf", FINANCE),
    }
    yield r
    r.env.engine.dispose()


def ids(rag, *names):
    return [rag.docs[n]["id"] for n in names]


def test_retrieval_finds_the_right_document_and_page(rag):
    top = rag.retriever.search("benchmark completed multi-step tasks baseline", top_k=3)[0]
    assert (top.filename, top.page) == ("orchestra.pdf", 2) and "87%" in top.text
    top = rag.retriever.search("sourdough starter fermentation oven", top_k=1)[0]
    assert (top.filename, top.page) == ("cooking.pdf", 2)
    top = rag.retriever.search("hosting contract operating expenses budget", top_k=1)[0]
    assert top.filename == "finance.pdf"


def test_scores_are_sorted_bounded_and_top_k_is_respected(rag):
    res = rag.retriever.search("graph state agents checkpointing", top_k=4)
    assert len(res) == 4 and [c.score for c in res] == sorted((c.score for c in res), reverse=True)
    assert all(-1.0001 <= c.score <= 1.0001 for c in res)
    assert len(rag.retriever.search("graph state agents", top_k=1)) == 1
    assert len(rag.retriever.search("graph state agents", top_k=500)) <= 100


def test_metadata_filters_by_document_and_page_range(rag):
    only = rag.retriever.search("graph state agents", top_k=10, document_ids=ids(rag, "cooking"))
    assert {c.filename for c in only} == {"cooking.pdf"}
    two = rag.retriever.search("agents", top_k=10, document_ids=ids(rag, "orchestra", "finance"))
    assert {c.filename for c in two} <= {"orchestra.pdf", "finance.pdf"}
    p3 = rag.retriever.search(
        "risks complexity budget limits", top_k=10, document_ids=ids(rag, "orchestra"), page_min=3
    )
    assert p3 and {c.page for c in p3} == {3}
    mid = rag.retriever.search(
        "results", top_k=10, document_ids=ids(rag, "orchestra"), page_min=1, page_max=2
    )
    assert {c.page for c in mid} <= {1, 2} and 3 not in {c.page for c in mid}
    assert rag.retriever.search("anything", document_ids=[]) == []
    assert rag.retriever.search("anything", document_ids=["not-a-real-id"]) == []


def test_documents_that_are_not_ready_never_appear_and_cache_invalidates(rag):
    q = "sourdough starter fermentation"
    assert rag.retriever.search(q, 1)[0].filename == "cooking.pdf"  # warms the cache
    db = rag.env.SF()
    d = db.get(UploadedDocument, rag.docs["cooking"]["id"])
    d.status = "processing"
    db.commit()
    db.close()
    assert "cooking.pdf" not in {c.filename for c in rag.retriever.search(q, 10)}
    db = rag.env.SF()
    db.get(UploadedDocument, rag.docs["cooking"]["id"]).status = "ready"
    db.commit()
    db.close()
    assert rag.retriever.search(q, 1)[0].filename == "cooking.pdf"


def test_deleted_documents_disappear_immediately(rag):
    assert rag.retriever.search("hosting contract", 1)[0].filename == "finance.pdf"
    assert rag.service.delete(rag.docs["finance"]["id"]) is True
    assert "finance.pdf" not in {c.filename for c in rag.retriever.search("hosting contract", 10)}
    db = rag.env.SF()
    assert db.query(DocumentChunk).filter_by(document_id=rag.docs["finance"]["id"]).count() == 0
    db.close()


def test_other_embedding_models_are_excluded_until_reindexed(rag, tmp_path):
    other = HashingEmbedder(256)  # a different vector space (different dimension and name)
    r2 = Retriever(SqlIndex(rag.env.SF, other.name), other, rag.env.SF)
    assert r2.search("graph state agents", 5) == []  # nothing embedded with this model yet
    for d in rag.docs.values():
        rag.service.embedder = other
        rag.service.index = r2.index
        rag.service.reindex(d["id"])
    res = r2.search("graph state agents", 3)
    assert res and res[0].filename == "orchestra.pdf"
    db = rag.env.SF()
    assert {d.embedding_model for d in db.scalars(select(UploadedDocument))} == {"hash-256-v1"}
    assert rag.retriever.search("graph state agents", 5) == []  # old space no longer matches
    db.close()


def test_get_chunks_preserves_requested_order_and_hides_missing_ids(rag):
    got = rag.retriever.search("graph state", 3)
    wanted = [got[2].chunk_id, got[0].chunk_id, "nope:1"]
    out = rag.retriever.get_chunks(wanted)
    assert [c.chunk_id for c in out] == [got[2].chunk_id, got[0].chunk_id] and all(
        c.score is None for c in out
    )


def test_embedding_failures_surface_as_retrieval_errors_with_retryability(rag):
    class Down:
        name = rag.embedder.name

        def embed(self, texts):
            raise EmbeddingError("provider returned HTTP 503", retryable=True)

    r = Retriever(rag.index, Down(), rag.env.SF)
    with pytest.raises(RetrievalError) as ei:
        r.search("anything", 3)
    assert ei.value.retryable is True


def test_lexical_reranker_promotes_exact_term_matches():
    def c(i, score, text):
        return Chunk(
            chunk_id=f"c{i}", document_id="d", filename="f.pdf", page=1, score=score, text=text
        )

    chunks = [
        c(1, 0.90, "General discussion of limits and performance across many systems and topics."),
        c(2, 0.80, "The ZX-90 controller exposes ZX-90 limits and ZX-90 calibration settings."),
        c(3, 0.70, "Unrelated notes about lunch."),
    ]
    out = LexicalReranker(0.5).rerank("ZX-90 limits", chunks)
    assert out[0].chunk_id == "c2" and out[-1].chunk_id == "c3"
    same = LexicalReranker().rerank("anything", [c(1, 0.5, "a b"), c(2, 0.5, "c d")])
    assert [x.chunk_id for x in same] == ["c1", "c2"]  # equal scores: stable, no crash
    assert LexicalReranker().rerank("q", chunks[:1]) == chunks[:1]


def test_reranking_retriever_returns_top_k_after_overfetch(tmp_path):
    r = make_rag(tmp_path, rerank=True)
    r.upload("orchestra.pdf", ORCHESTRA)
    r.upload("cooking.pdf", COOKING)
    res = r.retriever.search("spaghetti garlic olive oil", top_k=2)
    assert len(res) == 2 and res[0].filename == "cooking.pdf"
    r.env.engine.dispose()


# --------------------------------------------------------------------------- Chroma (optional)
chromadb = pytest.importorskip("chromadb", reason="optional dependency")


def test_chroma_index_matches_the_sql_index_and_supports_filters(rag, tmp_path):
    chroma = ChromaIndex(str(tmp_path / "chroma"), rag.embedder.name, rag.env.SF)
    assert chroma.rebuild() == sum(d["chunk_count"] for d in rag.docs.values())
    vec = rag.embedder.embed(["graph state agents checkpointing"])[0]
    a = rag.index.query(vec, 5)
    b = chroma.query(vec, 5)
    strong = [h for h in a if h.score > 0.05]  # near-zero scores are ties: their order is arbitrary
    assert len(strong) >= 2
    assert [h.chunk_id for h in strong] == [h.chunk_id for h in b][: len(strong)]
    by_id = {h.chunk_id: h.score for h in b}
    assert np.allclose([h.score for h in strong], [by_id[h.chunk_id] for h in strong], atol=1e-3)
    only = chroma.query(vec, 10, document_ids=ids(rag, "cooking"))
    assert {h.document_id for h in only} == set(ids(rag, "cooking"))
    p3 = chroma.query(vec, 10, document_ids=ids(rag, "orchestra"), page_min=3)
    assert p3 and {h.page for h in p3} == {3}
    assert chroma.query(vec, 5, document_ids=[]) == []


def test_chroma_stays_in_sync_through_service_hooks_and_self_heals(rag, tmp_path):
    chroma = ChromaIndex(str(tmp_path / "chroma2"), rag.embedder.name, rag.env.SF)
    chroma.rebuild()
    rag.service.index = chroma  # ingestion/deletion now maintain the derived index
    new = rag.upload("extra.pdf", ["Brand new document about kubernetes autoscaling and pods."])
    vec = rag.embedder.embed(["kubernetes autoscaling pods"])[0]
    assert chroma.query(vec, 1)[0].document_id == new["id"]
    rag.service.delete(new["id"])
    assert new["id"] not in {h.document_id for h in chroma.query(vec, 10)}
    chroma._col.delete(where={"document_id": rag.docs["finance"]["id"]})  # simulate drift
    assert chroma.ensure_synced() is True and chroma.ensure_synced() is False
    assert rag.docs["finance"]["id"] in {
        h.document_id for h in chroma.query(rag.embedder.embed(["hosting contract budget"])[0], 10)
    }
