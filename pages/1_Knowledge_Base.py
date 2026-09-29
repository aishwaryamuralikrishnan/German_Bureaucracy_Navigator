"""Knowledge Base page: inventory, rebuild with progress, and a retrieval playground."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navigator.config import PROCESSED_DIR, get_api_key, get_settings  # noqa: E402
from navigator.rag.ingest import ingest  # noqa: E402
from navigator.rag.retriever import get_retriever  # noqa: E402
from navigator.rag.vectorstore import document_inventory  # noqa: E402
from navigator.utils.errors import NavigatorError  # noqa: E402
from ui.theme import MUTED  # noqa: E402

st.set_page_config(page_title="Knowledge Base", page_icon="📚", layout="wide")
s = get_settings()

st.title("📚 Knowledge Base")

if not get_api_key():
    st.error(f"{s.llm.api_key_env} is not set — embeddings are created via OpenRouter.")
    st.stop()

# ----------------------------------------------------------------------------- status
try:
    retriever = get_retriever()
    count = retriever.count()
except NavigatorError as exc:
    st.error(str(exc))
    st.stop()

md_files = sorted(PROCESSED_DIR.rglob("*.md"))
def text_tile(col, label: str, value: str) -> None:
    """A metric-style tile for text values: st.metric truncates anything longer than a few characters."""
    col.markdown(
        f"<div style='font-size:0.875rem;color:{MUTED};margin-bottom:2px'>{label}</div>"
        f"<div style='font-size:1.25rem;font-weight:600;line-height:1.3;word-break:break-word'>{value}</div>",
        unsafe_allow_html=True,
    )


c1, c2, c3, c4 = st.columns([1, 1, 1.6, 1.4])
c1.metric("Markdown documents on disk", len(md_files))
c2.metric("Chunks indexed", count)
text_tile(c3, "Embedding model", s.embeddings.model)
text_tile(c4, "Reranker", (s.retrieval.reranker.model if s.retrieval.reranker.enabled else "off"))

# ----------------------------------------------------------------------------- rebuild
st.subheader("Rebuild index")
st.caption(
    "Reads every `data/processed/**/*.md`, chunks by headings, embeds via OpenRouter and stores in Chroma. "
    "To add official web pages, run `python scripts/fetch_sources.py` first."
)
if st.button("🔄 Rebuild knowledge base", type="primary"):
    bar = st.progress(0, text="Starting…")
    log_box = st.empty()
    lines: list[str] = []

    def _progress(stage: str, done: int, total: int, msg: str) -> None:
        base = {"load": (0.0, 0.03), "chunk": (0.03, 0.06), "ping": (0.06, 0.10), "embed": (0.10, 0.85), "write": (0.85, 1.0)}
        lo, hi = base.get(stage, (0.1, 0.85))
        frac = lo + (hi - lo) * (done / max(total, 1))
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)
        lines.append(f"{stage}: {msg}")
        log_box.code("\n".join(lines[-8:]))

    try:
        report = ingest(progress=_progress)
        bar.progress(1.0, text="Done")
        st.success(f"Indexed {report.indexed} chunks from {report.documents} documents with {report.embedding_model} ({report.embedding_dim} dims).")
        st.rerun()
    except (NavigatorError, RuntimeError) as exc:
        st.error(str(exc))

# ----------------------------------------------------------------------------- inventory
st.subheader("Indexed documents")
inventory = document_inventory(retriever.store) if count else []
if inventory:
    st.dataframe(pd.DataFrame(inventory), hide_index=True, width="stretch")
else:
    st.info("Nothing indexed yet.")

# ----------------------------------------------------------------------------- playground
st.subheader("Retrieval playground")
st.caption("See what hybrid search (BM25 + embeddings, fused with RRF, then reranked) returns before the LLM sees it. "
           "⚠️ weak = the passage shares no keyword with the question and does not stand out semantically, or the reranker scores it as not answering the question; "
           "when every returned passage is weak, the assistant says the knowledge base does not cover the question.")
q = st.text_input("Query", placeholder="e.g. Wohnungsgeberbestätigung — what is it?")
col1, col2 = st.columns([1, 3])
city = col1.selectbox("City filter", ["(none)"] + list(s.app.cities), accept_new_options=True)
if q:
    with st.spinner("Retrieving…"):
        try:
            chunks, trace = retriever.retrieve(q, city=None if city == "(none)" else city, k=5)
        except NavigatorError as exc:
            st.error(str(exc))
            st.stop()
    st.caption(
        f"Variants: {trace.variants} · BM25 hits {trace.bm25_hits} · dense hits {trace.dense_hits} · "
        f"fused {trace.fused} · reranker {trace.reranker} · filter {trace.jurisdiction_filter}"
    )
    if trace.weak_coverage:
        st.warning(f"Weak coverage: best standout z {trace.max_standout} (similarity {trace.max_similarity}) is below {trace.threshold} — {trace.threshold_source}. The agent would tell the user the knowledge base does not cover this.", icon="📭")
    for i, c in enumerate(chunks, 1):
        sim = (f"relevance {c.relevance:.2f}, " if c.relevance is not None else "") + (f"similarity {c.similarity:.2f}" if c.similarity is not None else "BM25-only") + (f", standout z {c.standout:.1f}" if c.standout is not None else "")
        with st.expander(f"[S{i}] {c.title} — {c.section}  ({sim}, fused {c.score:.4f}, {c.jurisdiction}, {c.language}){'  ⚠️ weak' if c.weak else ''}"):
            st.write(c.text)
            if c.source_url:
                st.markdown(f"[source]({c.source_url})")
