"""Heading-aware chunking: split on markdown headers first, then by size."""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from navigator.config import get_settings

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _heading_path(meta: dict) -> str:
    return " > ".join(str(meta[h]) for _, h in HEADERS if meta.get(h))


def chunk_documents(docs: list[Document]) -> list[Document]:
    cfg = get_settings().chunking
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size_chars,
        chunk_overlap=cfg.chunk_overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in docs:
        sections = header_splitter.split_text(doc.page_content)
        for si, section in enumerate(sections):
            section_path = _heading_path(section.metadata)
            for pi, piece in enumerate(size_splitter.split_documents([section])):
                text = piece.page_content.strip()
                if len(text) < 40:
                    continue
                prefix = f"{doc.metadata.get('title', '')} > {section_path}".strip(" >")
                content = f"[{prefix}]\n{text}" if cfg.prefix_heading_path and prefix else text
                meta = {k: v for k, v in doc.metadata.items() if k not in {"h1", "h2", "h3"}}
                meta["section"] = section_path or doc.metadata.get("title", "")
                digest = hashlib.sha1(f"{meta.get('file')}|{si}|{pi}|{text[:80]}".encode("utf-8")).hexdigest()[:16]
                meta["chunk_id"] = digest
                chunks.append(Document(page_content=content, metadata=meta))
    return chunks
