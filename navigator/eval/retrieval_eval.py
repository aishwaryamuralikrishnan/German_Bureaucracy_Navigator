"""Retriever-side evaluation: hit-rate@k and MRR for four retrieval configurations.

The four configurations share the same query variants and embeddings per question, so the comparison
isolates the retrieval method itself:

    dense          embedding similarity only
    bm25           keyword (BM25) only
    hybrid         BM25 + dense fused with Reciprocal Rank Fusion, no reranker
    hybrid_rerank  the production pipeline: hybrid + cross-encoder reranker

A question is a "hit" when any of its expected documents appears in the top-k; its reciprocal rank is
1 / (rank of the first expected document), 0 when absent.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from navigator.core.schemas import RetrievedChunk
from navigator.eval.questions import EvalQuestion
from navigator.rag.hybrid import dense_search_by_vector, reciprocal_rank_fusion
from navigator.rag.retriever import KnowledgeBaseRetriever

CONFIGS: dict[str, str] = {
    "dense": "dense only",
    "bm25": "BM25 only",
    "hybrid": "hybrid (RRF)",
    "hybrid_rerank": "hybrid + reranker",
}

Progress = Callable[[str, int, int, str], None]


@dataclass
class QuestionRetrieval:
    id: str
    group: str
    expected_docs: list[str]
    ranks: dict[str, int | None]            # config -> 1-based rank of the first expected doc, None = miss
    top_docs: dict[str, list[str]]           # config -> files of the returned passages, in order
    variants: list[str]
    seconds: float


@dataclass
class ConfigScore:
    config: str
    label: str
    questions: int
    hits: int
    hit_rate: float
    mrr: float


@dataclass
class RetrievalReport:
    k: int
    configs: list[ConfigScore]
    per_question: list[QuestionRetrieval]
    seconds: float
    reranker: str
    embedding_model: str
    skipped: list[str] = field(default_factory=list)   # questions without expected documents

    def to_dict(self) -> dict:
        return {
            "k": self.k, "seconds": round(self.seconds, 1), "reranker": self.reranker, "embedding_model": self.embedding_model,
            "configs": [asdict(c) for c in self.configs],
            "per_question": [asdict(q) for q in self.per_question],
            "skipped": self.skipped,
        }


def _first_rank(chunks: list[RetrievedChunk], expected: set[str]) -> int | None:
    for i, c in enumerate(chunks, start=1):
        if c.file in expected:
            return i
    return None


def rank_lists(retriever: KnowledgeBaseRetriever, question: str, city: str | None, k: int) -> tuple[dict[str, list[RetrievedChunk]], list[str]]:
    """Run the four configurations once for a question; returns config -> top-k chunks, plus the variants."""
    cfg = retriever.settings.retrieval
    allowed = retriever._jurisdictions(city)
    variants = retriever._expand(question)

    dense_lists: list[list[RetrievedChunk]] = []
    bm25_lists: list[list[RetrievedChunk]] = []
    for v in variants:
        qvec = retriever.embeddings.embed_query(v)
        dense_lists.append(dense_search_by_vector(retriever.store, qvec, cfg.dense_k, allowed))
        bm25_lists.append(retriever.bm25.search(v, cfg.bm25_k, allowed))

    dense_only = reciprocal_rank_fusion(dense_lists, k=cfg.rrf_k)[:k]
    bm25_only = reciprocal_rank_fusion(bm25_lists, k=cfg.rrf_k)[:k]
    fused = reciprocal_rank_fusion([x for pair in zip(bm25_lists, dense_lists) for x in pair], k=cfg.rrf_k)
    hybrid = fused[:k]
    hybrid_rerank = retriever.reranker.rerank(question, fused[: cfg.reranker.candidates], k)
    return {"dense": dense_only, "bm25": bm25_only, "hybrid": hybrid, "hybrid_rerank": hybrid_rerank}, variants


def evaluate_retrieval(
    questions: list[EvalQuestion],
    retriever: KnowledgeBaseRetriever,
    k: int = 5,
    progress: Progress | None = None,
) -> RetrievalReport:
    started = time.time()
    scored = [q for q in questions if q.expected_docs]
    skipped = [q.id for q in questions if not q.expected_docs]
    per_question: list[QuestionRetrieval] = []

    for i, q in enumerate(scored, start=1):
        t0 = time.time()
        city = _city_hint(q)
        lists, variants = rank_lists(retriever, q.question, city, k)
        expected = set(q.expected_docs)
        ranks = {name: _first_rank(chunks, expected) for name, chunks in lists.items()}
        per_question.append(QuestionRetrieval(
            id=q.id, group=q.group, expected_docs=q.expected_docs, ranks=ranks,
            top_docs={name: [c.file or c.title for c in chunks] for name, chunks in lists.items()},
            variants=variants, seconds=round(time.time() - t0, 2),
        ))
        if progress:
            progress("retrieval", i, len(scored), f"{q.id} " + " ".join(f"{n}={r if r else '—'}" for n, r in ranks.items()))

    configs: list[ConfigScore] = []
    for name, label in CONFIGS.items():
        ranks = [pq.ranks[name] for pq in per_question]
        hits = sum(1 for r in ranks if r is not None)
        n = len(ranks) or 1
        configs.append(ConfigScore(
            config=name, label=label, questions=len(ranks), hits=hits,
            hit_rate=round(hits / n, 4), mrr=round(sum(1 / r for r in ranks if r) / n, 4),
        ))
    return RetrievalReport(
        k=k, configs=configs, per_question=per_question, seconds=time.time() - started,
        reranker=retriever.reranker.name, embedding_model=getattr(retriever.embeddings, "model", "unknown"), skipped=skipped,
    )


def _city_hint(q: EvalQuestion) -> str | None:
    """The retriever receives the city the agent would pass — taken from the question's expected search arguments."""
    city = (q.expected_args.get("search_knowledge_base") or {}).get("city")
    return None if city in (None, "any") else str(city)
