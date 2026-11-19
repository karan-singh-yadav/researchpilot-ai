"""Harness for document/RAG tests: temp DB + upload dir, inline service, SQL index, retriever."""

from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.rag.embeddings import HashingEmbedder
from app.rag.index import SqlIndex
from app.rag.retriever import LexicalReranker, Retriever
from app.rag.service import DocumentService
from tests.graph_env import make_env
from tests.pdf_factory import ORCHESTRA, make_pdf, wrap

COOKING = [
    wrap(
        "Weeknight pasta recipes. Boil salted water, cook the spaghetti for nine minutes and toss it with "
        "garlic, olive oil and chilli flakes. Fresh basil and parmesan finish the dish."
    ),
    wrap(
        "Baking notes. Sourdough bread needs a mature starter, a long cold fermentation and a very hot "
        "oven. Flour, water and salt are the only ingredients required."
    ),
]
FINANCE = [
    wrap(
        "Quarterly budget review. Operating expenses rose by 4.2 percent because of cloud hosting costs. "
        "The finance team recommends renegotiating the annual hosting contract."
    )
]


@dataclass
class Rag:
    env: object
    settings: Settings
    embedder: object
    index: SqlIndex
    service: DocumentService
    retriever: Retriever

    def upload(self, name: str, pages: list[str], title: str | None = None) -> dict:
        doc, _ = self.service.upload(name, make_pdf(pages, title=title))
        return doc


def make_rag(
    tmp_path: Path, *, rerank=False, dim=512, executor="inline", embedder=None, **overrides
) -> Rag:
    env = make_env(tmp_path)
    settings = Settings(
        _env_file=None,
        upload_dir=str(tmp_path / "uploads"),
        chunk_size=overrides.pop("chunk_size", 400),
        chunk_overlap=overrides.pop("chunk_overlap", 60),
        run_executor=executor,
        embedding_dim=dim,
        **overrides,
    )
    emb = embedder or HashingEmbedder(dim)
    index = SqlIndex(env.SF, emb.name)
    service = DocumentService(env.SF, settings, emb, index, executor=executor)
    retriever = Retriever(index, emb, env.SF, LexicalReranker() if rerank else None)
    return Rag(env, settings, emb, index, service, retriever)


__all__ = ["ORCHESTRA", "COOKING", "FINANCE", "make_rag", "Rag"]
