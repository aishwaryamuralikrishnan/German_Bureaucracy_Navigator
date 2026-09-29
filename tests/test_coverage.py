import numpy as np

from navigator.rag.coverage import CorpusMatrix, query_stats


def _matrix(n=60, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(n, dim)).astype(np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    ids = [f"c{i}" for i in range(n)]
    return CorpusMatrix(ids, vec, {cid: i for i, cid in enumerate(ids)})


def test_relevant_chunk_stands_out_and_random_query_does_not():
    m = _matrix()
    # query almost identical to chunk c7 -> c7 must have a high z-score
    q = m.vectors[7] + 0.05 * np.random.default_rng(1).normal(size=m.vectors.shape[1]).astype(np.float32)
    st = query_stats(m, q)
    assert st is not None and st.z("c7") > 3.0
    # unrelated random query -> max z stays modest (below the strict threshold for most seeds)
    q2 = np.random.default_rng(99).normal(size=m.vectors.shape[1]).astype(np.float32)
    st2 = query_stats(m, q2)
    assert max(st2.z_by_id.values()) < 4.0


def test_small_corpus_returns_none():
    m = _matrix(n=10)
    assert query_stats(m, m.vectors[0]) is None
