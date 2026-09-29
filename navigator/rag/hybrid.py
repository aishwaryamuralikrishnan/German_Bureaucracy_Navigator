"""Hybrid retrieval: BM25 (keyword) + dense (embeddings) fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from navigator.core.schemas import RetrievedChunk
from navigator.rag.vectorstore import get_all_chunks

_TOKEN_RE = re.compile(r"[a-zA-ZäöüÄÖÜß0-9§]+")
_STOP = {
    # German
    "der", "die", "das", "und", "oder", "ist", "ich", "ein", "eine", "einen", "einem", "einer", "für", "mit", "von",
    "zu", "im", "in", "auf", "wie", "was", "wo", "wann", "wer", "wen", "wem", "welche", "welcher", "welches", "muss",
    "kann", "habe", "hat", "haben", "bin", "sind", "wird", "werden", "nicht", "auch", "noch", "nach", "bei", "aus",
    "dem", "den", "des", "sich", "mein", "meine", "meinen", "meiner", "dass", "wenn", "ob", "es", "sie", "er", "wir",
    # English
    "the", "a", "an", "and", "or", "is", "are", "am", "was", "were", "be", "been", "i", "to", "of", "for", "on", "at",
    "by", "my", "me", "we", "our", "you", "your", "they", "them", "their", "it", "its", "this", "that", "these", "those",
    "do", "does", "did", "how", "what", "when", "where", "who", "whom", "which", "why", "can", "could", "should", "would",
    "will", "may", "might", "must", "need", "want", "have", "has", "had", "get", "got", "there", "here", "about", "after",
    "before", "into", "from", "with", "without", "much", "many", "some", "any", "all", "not", "no", "yes", "if", "so",
    "as", "than", "then", "also", "just", "only", "very", "more", "most", "one", "two", "new",
}


def tokenize(text: str) -> list[str]:
    # "§ 17" -> "§17" so law paragraphs become a single strong keyword.
    text = re.sub(r"§\s*(\d+[a-z]?)", r"§\1", text.lower())
    return [t for t in _TOKEN_RE.findall(text) if t not in _STOP and len(t) > 1]


def _to_chunk(text: str, meta: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(meta.get("chunk_id", "")),
        text=text,
        title=str(meta.get("title", "untitled")),
        file=meta.get("file"),
        source_url=meta.get("source_url"),
        section=meta.get("section"),
        topic=meta.get("topic"),
        jurisdiction=meta.get("jurisdiction"),
        language=meta.get("language"),
        score=score,
    )


@dataclass
class BM25Index:
    ids: list[str]
    texts: list[str]
    metas: list[dict]
    model: BM25Okapi | None
    _token_sets: list[set[str]] = field(default_factory=list)

    @classmethod
    def build(cls, store: Chroma) -> "BM25Index":
        ids, texts, metas = get_all_chunks(store)
        model = BM25Okapi([tokenize(t) for t in texts]) if texts else None
        idx = cls(ids, texts, metas, model)
        idx._token_sets = [set(tokenize(t)) for t in texts]
        return idx

    def supported_ids(self, query: str, max_df_ratio: float = 0.2) -> set[str]:
        """Chunk ids that share at least one *discriminative* term with the query — a term that occurs in
        at most `max_df_ratio` of all chunks. Common words shared with half the corpus do not count."""
        if not self.texts:
            return set()
        n = len(self.texts)
        tokens = {t for t in tokenize(query) if len(t) > 2}
        out: set[str] = set()
        for t in tokens:
            hits = [i for i, ts in enumerate(self._token_sets) if t in ts]
            if hits and len(hits) / n <= max_df_ratio:
                out.update(self.metas[i].get("chunk_id", "") for i in hits)
        return out

    def search(self, query: str, k: int, allowed_jurisdictions: set[str] | None = None) -> list[RetrievedChunk]:
        if not self.model:
            return []
        scores = self.model.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[RetrievedChunk] = []
        for i in order:
            if scores[i] <= 0:
                break
            if allowed_jurisdictions and self.metas[i].get("jurisdiction") not in allowed_jurisdictions:
                continue
            out.append(_to_chunk(self.texts[i], self.metas[i], float(scores[i])))
            if len(out) >= k:
                break
        return out


def dense_search(store: Chroma, query: str, k: int, allowed_jurisdictions: set[str] | None = None) -> list[RetrievedChunk]:
    flt = {"jurisdiction": {"$in": sorted(allowed_jurisdictions)}} if allowed_jurisdictions else None
    results = store.similarity_search_with_score(query, k=k, filter=flt)
    return _dense_results(results)


def dense_search_by_vector(store: Chroma, query_vec: list[float], k: int, allowed_jurisdictions: set[str] | None = None) -> list[RetrievedChunk]:
    """Same as dense_search but with a pre-computed query embedding (avoids embedding twice)."""
    flt = {"jurisdiction": {"$in": sorted(allowed_jurisdictions)}} if allowed_jurisdictions else None
    results = store.similarity_search_by_vector_with_relevance_scores(list(query_vec), k=k, filter=flt)
    # relevance = 1 - cosine distance for this collection, i.e. already a similarity
    return [_to_chunk(doc.page_content, doc.metadata, round(float(rel), 4)).model_copy(update={"similarity": round(float(rel), 4)}) for doc, rel in results]


def _dense_results(results) -> list[RetrievedChunk]:
    # Chroma returns cosine *distance*; convert to similarity for readability.
    out = []
    for doc, dist in results:
        sim = round(1.0 - float(dist), 4)
        out.append(_to_chunk(doc.page_content, doc.metadata, sim).model_copy(update={"similarity": sim}))
    return out


def reciprocal_rank_fusion(result_lists: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    fused: dict[str, float] = {}
    keep: dict[str, RetrievedChunk] = {}
    best_sim: dict[str, float] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            keep.setdefault(chunk.chunk_id, chunk)
            if chunk.similarity is not None:
                best_sim[chunk.chunk_id] = max(best_sim.get(chunk.chunk_id, -1.0), chunk.similarity)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [
        keep[cid].model_copy(update={"score": round(score, 5), "similarity": best_sim.get(cid)})
        for cid, score in ordered
    ]
