import io
import json
import re
import sys
import types

import httpx
import numpy as np
import pytest
from pypdf import PdfReader, PdfWriter

from app.config import Settings
from app.rag.chunking import chunk_pages
from app.rag.embeddings import (
    EmbeddingError,
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from app.rag.pdf import PdfError, PdfPage, extract_pdf, validate_pdf_header
from tests.pdf_factory import ORCHESTRA, make_pdf, wrap


# ------------------------------------------------------------------------------ PDF
def test_extracts_text_with_correct_page_numbers_and_metadata():
    c = extract_pdf(
        make_pdf(["alpha page one text here", "", "gamma page three text here"], title="My Doc")
    )
    assert c.page_count == 3 and [p.number for p in c.pages] == [
        1,
        3,
    ]  # blank page 2 skipped, numbers kept
    assert "gamma" in c.pages[1].text and c.metadata == {"title": "My Doc"}


@pytest.mark.parametrize(
    "data", [b"", b"hello world", b"<html>%not pdf</html>", b"PK\x03\x04zipfile"]
)
def test_non_pdf_bytes_are_rejected_by_header(data):
    with pytest.raises(PdfError) as ei:
        validate_pdf_header(data)
    assert ei.value.code == "not_pdf"


def test_corrupt_and_truncated_pdfs_give_a_safe_error():
    good = make_pdf(ORCHESTRA)
    for bad in (
        good[: len(good) // 3],
        b"%PDF-1.4\n" + b"\x00garbage" * 50,
        b"%PDF-1.7\n1 0 obj << /Type /Nope",
    ):
        with pytest.raises(PdfError) as ei:
            extract_pdf(bad)
        assert ei.value.code in ("corrupt", "no_text") and "Traceback" not in ei.value.message


def test_encrypted_pdf_is_rejected_but_empty_password_pdf_is_readable():
    def encrypted(password: str) -> bytes:
        w = PdfWriter()
        for pg in PdfReader(io.BytesIO(make_pdf(ORCHESTRA))).pages:
            w.add_page(pg)
        w.encrypt(user_password=password, owner_password="owner")
        buf = io.BytesIO()
        w.write(buf)
        return buf.getvalue()

    with pytest.raises(PdfError) as ei:
        extract_pdf(encrypted("secret"))
    assert ei.value.code == "encrypted"
    assert extract_pdf(encrypted("")).page_count == 3


def test_limits_on_pages_and_text_and_scanned_documents():
    many = make_pdf([f"page {i} has some text" for i in range(6)])
    with pytest.raises(PdfError) as ei:
        extract_pdf(many, max_pages=5)
    assert ei.value.code == "too_many_pages"
    with pytest.raises(PdfError) as ei:
        extract_pdf(make_pdf([wrap("word " * 400)] * 3), max_chars=1000)
    assert ei.value.code == "too_much_text"
    with pytest.raises(PdfError) as ei:  # image-only/scanned PDFs have no text layer
        extract_pdf(make_pdf(["", ""]))
    assert ei.value.code == "no_text" and "OCR" in ei.value.message


def test_text_cleanup_removes_control_chars_and_rejoins_hyphenated_words():
    c = extract_pdf(make_pdf(["inter-\nnational standards\n\n\n\nnext   paragraph"]))
    assert "international standards" in c.pages[0].text and "\n\n\n" not in c.pages[0].text


# --------------------------------------------------------------------------- chunking
def pages(*texts):
    return [PdfPage(i + 1, t) for i, t in enumerate(texts)]


def test_chunks_never_span_pages_and_keep_their_page_numbers():
    text = " ".join(f"Sentence number {i} says something useful." for i in range(60))
    chunks = chunk_pages(
        pages(text, "Short second page with enough words to keep around."), 300, 50
    )
    assert {c.page for c in chunks} == {1, 2} and chunks[-1].page == 2
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(len(c.text) <= 600 for c in chunks)
    first_page = [c for c in chunks if c.page == 1]
    assert len(first_page) > 3


def test_overlap_repeats_context_between_consecutive_chunks():
    text = " ".join(f"Fact{i} is documented." for i in range(80))
    with_overlap = chunk_pages(pages(text), 300, 100)
    last_fact = re.findall(r"Fact\d+", with_overlap[0].text)[
        -1
    ]  # unique token at the end of chunk 0
    assert last_fact in with_overlap[1].text  # ...is repeated at the start of chunk 1
    no_overlap = chunk_pages(pages(text), 300, 0)
    last_fact = re.findall(r"Fact\d+", no_overlap[0].text)[-1]
    assert last_fact not in no_overlap[1].text


def test_very_long_tokens_and_sentences_are_hard_split_and_tiny_tails_folded():
    chunks = chunk_pages(pages("x" * 2500), 400, 0)
    assert len(chunks) == 7 and all(len(c.text) <= 400 for c in chunks)
    sentence = "word " * 300
    assert all(len(c.text) <= 400 for c in chunk_pages(pages(sentence), 400, 50))
    folded = chunk_pages(pages(("Alpha beta gamma. " * 20) + "End."), 120, 0, min_chunk=60)
    assert all(len(c.text) >= 20 for c in folded)


def test_chunking_is_deterministic_and_validates_parameters():
    text = wrap("Deterministic text for chunking. " * 40)
    assert chunk_pages(pages(text), 300, 60) == chunk_pages(pages(text), 300, 60)
    with pytest.raises(ValueError):
        chunk_pages(pages(text), 50, 0)
    with pytest.raises(ValueError):
        chunk_pages(pages(text), 300, 200)


# --------------------------------------------------------------------------- embeddings
def test_hashing_embedder_is_deterministic_unit_norm_and_ranks_by_term_overlap():
    e = HashingEmbedder(512)
    docs = [
        "Orchestra-X organizes agents as nodes in a directed graph with typed shared state",
        "The cafeteria menu includes soup, salad and several vegetarian pasta options",
        "Agent frameworks use graphs of nodes to coordinate work with shared state",
    ]
    m = e.embed(docs)
    assert m.shape == (3, 512) and np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(m, e.embed(docs))  # deterministic across calls
    q = e.embed(["how do agent graphs share state between nodes"])[0]
    assert list(np.argsort(-(m @ q))[:2]) in ([2, 0], [0, 2]) and (m @ q).argmin() == 1
    assert e.embed([]).shape == (0, 512)
    assert np.allclose(e.embed([""]), 0.0)  # no tokens -> zero vector (matches nothing)


def test_hashing_dimension_is_part_of_the_vector_space_name():
    assert (
        HashingEmbedder(256).name != HashingEmbedder(512).name
        and HashingEmbedder(64).embed(["x y"]).shape[1] == 64
    )


def test_openai_embedder_batches_orders_by_index_and_normalises():
    seen = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        seen.append((req.headers["authorization"], body["model"], len(body["input"])))
        data = [
            {"index": i, "embedding": [float(len(t)), 1.0, 0.0]}
            for i, t in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})  # out of order on purpose

    emb = OpenAIEmbedder(
        model="m",
        api_key="sk-x",
        base_url="https://e.test/v1/",
        batch_size=2,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    out = emb.embed(["a", "bb", "ccc", "dddd", "eeeee"])
    assert [n for _, _, n in seen] == [2, 2, 1] and seen[0][:2] == ("Bearer sk-x", "m")
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    first = list(out[:, 0])  # texts get longer, so this component strictly grows if order is right
    assert first == sorted(first) and len(set(first)) == 5
    assert emb.name == "openai:m"


@pytest.mark.parametrize("status,retryable", [(429, True), (503, True), (401, False), (400, False)])
def test_openai_embedder_errors_are_classified_and_never_leak_the_key(status, retryable):
    emb = OpenAIEmbedder(
        model="m",
        api_key="SECRETKEY",
        base_url="https://e.test/v1",
        retries=1,
        sleep=lambda s: None,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(status, text="SECRETKEY"))
        ),
    )
    with pytest.raises(EmbeddingError) as ei:
        emb.embed(["x"])
    assert ei.value.retryable is retryable and "SECRETKEY" not in str(ei.value)


def test_openai_embedder_retries_then_succeeds_and_rejects_malformed_responses():
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    sleeps = []
    emb = OpenAIEmbedder(
        model="m",
        api_key="k",
        base_url="https://e.test/v1",
        retries=2,
        sleep=sleeps.append,
        client=httpx.Client(transport=httpx.MockTransport(flaky)),
    )
    assert emb.embed(["x"]).shape == (1, 2) and calls["n"] == 3 and sleeps == [0.5, 1.0]
    for payload in (
        {"nope": 1},
        {"data": []},
        {"data": [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [1.0, 2.0]}]},
    ):
        bad = OpenAIEmbedder(
            model="m",
            api_key="k",
            base_url="https://e.test/v1",
            client=httpx.Client(
                transport=httpx.MockTransport(lambda r, p=payload: httpx.Response(200, json=p))
            ),
        )
        with pytest.raises(EmbeddingError):
            bad.embed(["x", "y"])


def test_sentence_transformers_adapter_calls_the_library_correctly(monkeypatch):
    class FakeModel:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, batch_size, normalize_embeddings):
            assert normalize_embeddings is True and batch_size == 32
            return np.array([[3.0, 4.0] for _ in texts])

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel)
    )
    e = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    out = e.embed(["a", "b"])
    assert e.name == "st:all-MiniLM-L6-v2" and np.allclose(out[0], [0.6, 0.8])
    monkeypatch.delitem(sys.modules, "sentence_transformers")
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # simulates "not installed"
    with pytest.raises(EmbeddingError, match="not installed"):
        SentenceTransformerEmbedder("x")


def test_build_embedder_validates_configuration():
    assert isinstance(build_embedder(Settings(_env_file=None)), HashingEmbedder)
    assert build_embedder(Settings(_env_file=None, embedding_dim=128)).name == "hash-128-v1"
    with pytest.raises(EmbeddingError):
        build_embedder(Settings(_env_file=None, embedding_provider="openai", embedding_model="m"))
    with pytest.raises(EmbeddingError):
        build_embedder(Settings(_env_file=None, embedding_provider="openai", embedding_api_key="k"))
    assert (
        build_embedder(
            Settings(
                _env_file=None,
                embedding_provider="openai",
                embedding_model="m",
                embedding_base_url="http://localhost:11434/v1",
            )
        ).name
        == "openai:m"
    )
    with pytest.raises(EmbeddingError):
        build_embedder(Settings(_env_file=None, embedding_provider="magic"))
