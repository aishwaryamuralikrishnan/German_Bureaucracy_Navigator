"""Chunking + hybrid retrieval end-to-end using a hashing embedder (no network)."""

import pytest

from navigator.rag.chunking import chunk_documents
from navigator.rag.hybrid import reciprocal_rank_fusion, tokenize
from navigator.rag.loaders import load_processed_documents, parse_front_matter
from navigator.core.schemas import RetrievedChunk


def test_front_matter_parsing():
    meta, body = parse_front_matter("---\ntitle: T\ntopic: taxes\n---\n\n# Hi\nbody")
    assert meta == {"title": "T", "topic": "taxes"} and body.startswith("# Hi")
    meta2, body2 = parse_front_matter("no front matter")
    assert meta2 == {} and body2 == "no front matter"


def test_seed_documents_chunk_with_metadata():
    docs = load_processed_documents()
    assert len(docs) >= 10
    chunks = chunk_documents(docs)
    assert len(chunks) > len(docs)
    for c in chunks:
        assert c.metadata["chunk_id"] and c.metadata["title"] and c.metadata["section"]
        assert c.metadata["jurisdiction"] in {"federal", "berlin", "munich", "hamburg", "frankfurt", "cologne", "stuttgart", "rostock"}
        assert c.page_content.startswith("[")  # heading path prefix
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)  # unique ids


def test_tokenizer_handles_umlauts_and_paragraph_signs():
    assert "anmeldung" in tokenize("Die Anmeldung § 17 BMG")
    assert "§17" in tokenize("§ 17") and "§18g" in tokenize("§ 18g AufenthG")
    assert "der" not in tokenize("der Termin")


def test_rrf_prefers_items_ranked_in_both_lists():
    a = RetrievedChunk(chunk_id="a", text="", title="a")
    b = RetrievedChunk(chunk_id="b", text="", title="b")
    c = RetrievedChunk(chunk_id="c", text="", title="c")
    fused = reciprocal_rank_fusion([[a, b], [b, c]], k=60)
    assert [x.chunk_id for x in fused][0] == "b"


@pytest.mark.slow
def test_ingest_and_retrieve_with_fake_embeddings(tmp_path, monkeypatch):
    from fakes import HashEmbeddings
    from navigator.config import get_settings
    from navigator.rag import vectorstore as vs
    from navigator.rag.ingest import ingest
    from navigator.rag.rerank import NoOpReranker
    from navigator.rag.retriever import KnowledgeBaseRetriever

    # Point Chroma at a temp dir so the test never touches the real index.
    monkeypatch.setattr(vs, "resolve_path", lambda _rel: tmp_path / "chroma")
    get_settings.cache_clear()

    report = ingest(embeddings=HashEmbeddings())
    assert report.indexed == report.chunks > 0

    r = KnowledgeBaseRetriever(embeddings=HashEmbeddings(), query_expansion=False, reranker=NoOpReranker())
    chunks, trace = r.retrieve("How many days do I have to register my address after moving?", city="Berlin", k=3)
    assert chunks and "Anmeldung" in chunks[0].title
    assert trace.jurisdiction_filter == ["berlin", "federal"]
    assert "auto" in trace.threshold_source and trace.max_standout is not None
    assert not trace.weak_coverage  # on-topic: keyword overlap alone rules out "weak"
    # off-topic with no shared keywords and no semantic standout -> weak
    _, tr2 = r.retrieve("who won the football world cup", k=3)
    assert tr2.weak_coverage
