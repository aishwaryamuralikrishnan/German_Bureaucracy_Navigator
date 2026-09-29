"""End-to-end retrieval: (query expansion) -> BM25 + dense -> RRF -> rerank -> top-k."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from navigator.config import REFERENCE_DIR, get_settings
from navigator.core.schemas import RetrievedChunk
from navigator.rag.coverage import CorpusMatrix, query_stats
from navigator.rag.embeddings import OpenRouterEmbeddings
from navigator.rag.hybrid import BM25Index, dense_search_by_vector, reciprocal_rank_fusion
from navigator.rag.rerank import get_reranker
from navigator.rag.vectorstore import collection_count, get_vectorstore
from navigator.utils.errors import KnowledgeBaseError
from navigator.utils.logging import get_logger, log_event

log = get_logger(__name__)

CITY_JURISDICTION = {
    "berlin": "berlin", "munich": "munich", "münchen": "munich", "hamburg": "hamburg",
    "frankfurt": "frankfurt", "cologne": "cologne", "köln": "cologne", "stuttgart": "stuttgart",
    "rostock": "rostock",
}


@lru_cache(maxsize=1)
def glossary_lines() -> str:
    with (REFERENCE_DIR / "glossary.yaml").open("r", encoding="utf-8") as fh:
        terms = yaml.safe_load(fh)["terms"]
    return "\n".join(f"- {t['de']} = {t['en']}" for t in terms)


EXPANSION_PROMPT = """You rewrite a user's question about German bureaucracy into search queries.
Return a JSON array of at most {n} short search strings:
- one in German using the official administrative terms (see glossary),
- one in English,
- optionally one more with synonyms or the relevant law paragraph (e.g. "§ 18g AufenthG").
Glossary (German = English):
{glossary}
Return ONLY the JSON array."""


@dataclass
class RetrievalTrace:
    variants: list[str] = field(default_factory=list)
    bm25_hits: int = 0
    dense_hits: int = 0
    fused: int = 0
    reranker: str = "none"
    jurisdiction_filter: list[str] | None = None
    max_similarity: float | None = None
    max_standout: float | None = None   # best robust z-score among returned passages
    weak_coverage: bool = False         # no returned passage stands out from the corpus
    threshold: float = 0.0              # z-score threshold (or absolute similarity if overridden)
    threshold_source: str = ""
    min_relevance: float | None = None  # reranker relevance below which a passage is weak (None = no reranker)
    max_relevance: float | None = None  # best reranker relevance among returned passages


class KnowledgeBaseRetriever:
    def __init__(self, embeddings=None, query_expansion: bool | None = None, reranker=None) -> None:
        self.settings = get_settings()
        self.embeddings = embeddings or OpenRouterEmbeddings()
        self._expansion_override = query_expansion
        self.store = get_vectorstore(self.embeddings)
        self.bm25 = BM25Index.build(self.store)
        self.matrix = CorpusMatrix.from_store(self.store)
        self.reranker = reranker or get_reranker()
        self._expander = None

    # ------------------------------------------------------------------ helpers
    def count(self) -> int:
        return collection_count(self.store)

    def refresh(self) -> None:
        """Rebuild the in-memory BM25 index and embedding matrix after ingestion."""
        self.bm25 = BM25Index.build(self.store)
        self.matrix = CorpusMatrix.from_store(self.store)

    def _expand(self, query: str) -> list[str]:
        cfg = self.settings.retrieval.query_expansion
        enabled = cfg.enabled if self._expansion_override is None else self._expansion_override
        if not enabled:
            return [query]
        try:
            if self._expander is None:
                from navigator.agent.llm import get_chat_model

                self._expander = get_chat_model(self.settings.llm.small_model, temperature=0.0)
            msg = self._expander.invoke(
                [
                    SystemMessage(content=EXPANSION_PROMPT.format(n=cfg.max_variants, glossary=glossary_lines())),
                    HumanMessage(content=query),
                ]
            )
            text = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            match = re.search(r"\[.*\]", text, flags=re.S)
            variants = json.loads(match.group(0)) if match else []
            variants = [v.strip() for v in variants if isinstance(v, str) and v.strip()]
            out = [query] + [v for v in variants if v.lower() != query.lower()]
            return out[: cfg.max_variants + 1]
        except Exception as exc:  # expansion is best-effort
            log.warning(f"query expansion failed: {exc}")
            return [query]

    @staticmethod
    def _jurisdictions(city: str | None) -> set[str] | None:
        if not city:
            return None
        j = CITY_JURISDICTION.get(city.strip().lower())
        return {"federal", j} if j else None

    # ------------------------------------------------------------------ main entry
    def retrieve(self, query: str, city: str | None = None, topic: str | None = None, k: int | None = None) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        if self.count() == 0:
            raise KnowledgeBaseError("The knowledge base is empty. Run `python scripts/ingest.py` first.")
        cfg = self.settings.retrieval
        k = k or cfg.final_k
        trace = RetrievalTrace(reranker=self.reranker.name)
        allowed = self._jurisdictions(city)
        trace.jurisdiction_filter = sorted(allowed) if allowed else None

        variants = self._expand(query)
        trace.variants = variants

        result_lists: list[list[RetrievedChunk]] = []
        best_z: dict[str, float] = {}
        bm25_ids: set[str] = set()
        for v in variants:
            qvec = self.embeddings.embed_query(v)
            bm = self.bm25.search(v, cfg.bm25_k, allowed)
            de = dense_search_by_vector(self.store, qvec, cfg.dense_k, allowed)
            trace.bm25_hits += len(bm)
            trace.dense_hits += len(de)
            bm25_ids.update(self.bm25.supported_ids(v))
            result_lists.extend([bm, de])
            stats = query_stats(self.matrix, qvec)
            if stats:
                for cid, z in stats.z_by_id.items():
                    if z > best_z.get(cid, float("-inf")):
                        best_z[cid] = z

        fused = reciprocal_rank_fusion(result_lists, k=cfg.rrf_k)
        if topic:
            preferred = [c for c in fused if c.topic == topic]
            fused = preferred + [c for c in fused if c.topic != topic]
        trace.fused = len(fused)

        candidates = fused[: cfg.reranker.candidates]
        final = self.reranker.rerank(query, candidates, k)

        # Relevance flags — deliberately conservative: we only ever claim "weak" when two independent
        # signals agree. Default ("auto"): a passage is weak if it shares NO discriminative keyword with any
        # query variant (see BM25Index.supported_ids) AND it does not stand out semantically from the corpus for this query
        # (robust z-score below retrieval.coverage_z). A number in retrieval.min_similarity switches to
        # an absolute cosine cut-off instead. Whether passages actually answer the question is left to the model.
        # Third signal (when a reranker ran): the cross-encoder read question and passage together. A passage
        # it scores below min_relevance does not answer the question even if it shares a keyword with it.
        override = cfg.get("min_similarity", "auto")
        z_min = float(cfg.get("coverage_z", 2.5))
        rel_min = float(cfg.reranker.get("min_relevance", 0.0) or 0.0)
        has_rel = any(c.relevance is not None for c in final)
        flagged: list[RetrievedChunk] = []
        for c in final:
            z = best_z.get(c.chunk_id)
            if isinstance(override, (int, float)):
                weak = c.similarity is not None and c.similarity < float(override)
            else:
                weak = (c.chunk_id not in bm25_ids) and (z is not None and z < z_min)
            if has_rel and c.relevance is not None and c.relevance < rel_min:
                weak = True
            flagged.append(c.model_copy(update={"weak": weak, "standout": round(z, 2) if z is not None else None}))
        final = flagged
        if has_rel:
            trace.min_relevance = rel_min
            rels = [c.relevance for c in final if c.relevance is not None]
            trace.max_relevance = max(rels) if rels else None
        if isinstance(override, (int, float)):
            trace.threshold, trace.threshold_source = float(override), "absolute similarity (settings.yaml)"
        else:
            trace.threshold, trace.threshold_source = z_min, ("no discriminative keyword overlap AND robust z-score below threshold (auto)"
                                                              + (f", OR reranker relevance below {rel_min}" if has_rel else ""))
        sims = [c.similarity for c in final if c.similarity is not None]
        zs = [c.standout for c in final if c.standout is not None]
        trace.max_similarity = max(sims) if sims else None
        trace.max_standout = max(zs) if zs else None
        trace.weak_coverage = bool(final) and all(c.weak for c in final)

        log_event(log, "retrieve", variants=len(variants), fused=len(fused), returned=len(final),
                  max_sim=trace.max_similarity, max_z=trace.max_standout, max_rel=trace.max_relevance, weak=trace.weak_coverage)
        return final, trace


_retriever: KnowledgeBaseRetriever | None = None


def get_retriever() -> KnowledgeBaseRetriever:
    """Process-wide singleton (Streamlit reruns the script; we keep the BM25 index warm)."""
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeBaseRetriever()
    return _retriever


def reset_retriever() -> None:
    global _retriever
    _retriever = None
