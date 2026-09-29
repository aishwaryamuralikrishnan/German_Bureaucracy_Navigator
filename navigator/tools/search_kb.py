from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from navigator.core.schemas import SourceRef, ToolResult
from navigator.rag.retriever import get_retriever
from navigator.utils.errors import ConfigError, ExternalServiceError, KnowledgeBaseError
from navigator.utils.logging import get_logger

log = get_logger(__name__)


class SearchInput(BaseModel):
    query: str = Field(description="The information need, in the user's words (German or English)")
    city: str | None = Field(None, description="City to prioritise city-specific sources, e.g. Berlin")
    topic: str | None = Field(
        None,
        description="Optional topic hint: anmeldung | residence_permits | students | taxes | health_insurance | banking | driving_licence | recognition | family_reunification | checklists",
    )
    k: int = Field(5, ge=1, le=8, description="Number of passages to return")


@tool("search_knowledge_base", args_schema=SearchInput)
def search_knowledge_base(query: str, city: str | None = None, topic: str | None = None, k: int = 5) -> str:
    """Search the curated knowledge base of official German bureaucracy information
    (registration, residence permits & visas, Blue Card, student visas, student health insurance, working as a
    student, blocked account, tax ID and tax classes, health insurance,
    banking, driving licence conversion, recognition of qualifications, family reunification,
    document checklists). Returns numbered passages [S1]..[Sk] with sources and a relevance flag.
    ALWAYS use this before answering factual questions about rules, procedures or documents, and cite
    only passages that actually support your sentence as [S1], [S2], ... If the result says
    coverage is weak, tell the user the knowledge base does not cover this specifically."""
    try:
        retriever = get_retriever()
        chunks, trace = retriever.retrieve(query, city=city, topic=topic, k=k)
    except (KnowledgeBaseError, ConfigError, ExternalServiceError) as exc:
        return ToolResult.failure("search_knowledge_base", str(exc)).to_json()
    except Exception as exc:  # anything else must still reach the model (and the evaluation) as a structured failure,
        # not as LangChain's generic "Error: … Please fix your mistakes" string that hides the cause
        log.exception("search_knowledge_base failed unexpectedly")
        return ToolResult.failure("search_knowledge_base", f"unexpected {type(exc).__name__}: {exc}").to_json()

    sources = []
    passages = []
    for i, c in enumerate(chunks, start=1):
        ref = f"S{i}"
        sources.append(
            SourceRef(
                ref=ref, title=c.title, url=c.source_url, section=c.section, snippet=c.text[:1200], score=c.score,
                jurisdiction=c.jurisdiction, language=c.language, topic=c.topic, similarity=c.similarity, standout=c.standout,
                relevance=c.relevance, weak=c.weak,
            )
        )
        passages.append({"ref": ref, "title": c.title, "section": c.section, "relevance": "weak" if c.weak else "ok",
                         "relevance_score": c.relevance, "text": c.text})

    warnings: list[str] = []
    if not chunks:
        warnings.append("No passages found. Say so rather than guessing.")
    elif trace.weak_coverage:
        warnings.append(
            "LOW COVERAGE: none of the passages is a strong match for this question. Tell the user the knowledge "
            "base has no specific information on this topic and point to the responsible authority (and its website). "
            "Then stop. Do NOT cite these passages, and do NOT state any amount, rate, duration, number of years, deadline "
            "or eligibility condition, and do NOT describe the procedure, application steps, required documents or who "
            "is eligible — not even approximately or 'typically' — because none of it can be checked against the knowledge base."
        )
    elif any(c.weak for c in chunks):
        warnings.append("Some passages are weak matches (relevance 'weak'); cite only the strong ones.")

    return ToolResult(
        tool="search_knowledge_base",
        data={
            "coverage": "none" if not chunks else "weak" if trace.weak_coverage else "ok",
            "passages": passages,
            "retrieval": {
                "query_variants": trace.variants,
                "bm25_hits": trace.bm25_hits,
                "dense_hits": trace.dense_hits,
                "fused_candidates": trace.fused,
                "reranker": trace.reranker,
                "jurisdiction_filter": trace.jurisdiction_filter,
                "max_similarity": trace.max_similarity,
                "max_standout": trace.max_standout,
                "max_relevance": trace.max_relevance,
                "min_relevance": trace.min_relevance,
                "weak_threshold": trace.threshold,
                "threshold_source": trace.threshold_source,
            },
        },
        sources=sources,
        warnings=warnings,
    ).to_json()
