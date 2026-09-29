"""Query-relative relevance ("does this passage stand out from the corpus?").

Every embedding model assigns some baseline similarity between a query and *any* chunk; the exact
level depends on the model and on how homogeneous the corpus is. Instead of a fixed similarity
cut-off, we compute, per query, the similarity to ALL chunks and measure how far each retrieved
chunk sits above the bulk, using robust statistics (median and MAD, so many relevant chunks do not
distort the baseline). A chunk whose robust z-score is below `coverage_z` is a weak match; if every
returned chunk is weak, the knowledge base does not cover the question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAD_SCALE = 1.4826  # makes MAD comparable to a standard deviation for normal data
MIN_CHUNKS_FOR_STATS = 25


@dataclass
class CorpusMatrix:
    ids: list[str]
    vectors: np.ndarray            # shape (n_chunks, dim), L2-normalised
    index: dict[str, int]          # chunk_id -> row

    @classmethod
    def from_store(cls, store) -> "CorpusMatrix":
        res = store.get(include=["embeddings"])
        ids = list(res.get("ids", []))
        emb = res.get("embeddings")
        if emb is None or len(ids) == 0:
            return cls([], np.zeros((0, 0)), {})
        vec = np.asarray(emb, dtype=np.float32)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return cls(ids, vec / norms, {cid: i for i, cid in enumerate(ids)})

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class QueryStats:
    median: float
    mad: float
    top: float
    z_by_id: dict[str, float]

    def z(self, chunk_id: str) -> float | None:
        return self.z_by_id.get(chunk_id)


def query_stats(matrix: CorpusMatrix, query_vec: list[float] | np.ndarray) -> QueryStats | None:
    """Similarity of the query to every chunk + robust z-score per chunk. None if corpus too small."""
    if len(matrix) < MIN_CHUNKS_FOR_STATS:
        return None
    q = np.asarray(query_vec, dtype=np.float32)
    n = np.linalg.norm(q)
    if n == 0:
        return None
    sims = matrix.vectors @ (q / n)
    med = float(np.median(sims))
    mad = float(np.median(np.abs(sims - med))) * MAD_SCALE
    if mad < 1e-6:
        mad = float(np.std(sims)) or 1e-6
    z = (sims - med) / mad
    return QueryStats(median=round(med, 4), mad=round(mad, 4), top=round(float(sims.max()), 4),
                      z_by_id={cid: float(z[i]) for cid, i in matrix.index.items()})
