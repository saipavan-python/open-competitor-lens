"""Turning a page into retrievable pieces, without losing where they came from.

Chunking is where a page stops being a page. Everything that makes a piece of
text useful in isolation — the company it is about, the source it came from, the
heading it sits under — has to survive the cut, or the analyst receives a
sentence about Beta's pricing with nothing tying it to Beta.

Two rules follow from that, and both are enforced here rather than left to
callers:

* **A heading never ends up in a different chunk from the text it introduces.**
  A chunk reading "Starter plans start at $10" with no heading is unattributable
  in a way that no amount of downstream care can repair.
* **Every chunk keeps the full metadata of its document.** Not a copy of the
  keys that happened to be set on the page — all of them. A chunk that silently
  lacks ``competitor`` fails every strict filter later, and does so by being
  invisible rather than by being wrong, which is the harder failure to notice.

The text splitting itself is delegated to :func:`app.utils.text.chunk_text`,
which already handles heading boundaries, paragraph integrity and the oversized
paragraph case. Reimplementing that here would mean two chunkers with two sets of
guarantees, one of which would be the one nobody tested.
"""

from __future__ import annotations

import re

from app.schemas.document import Chunk, Document
from app.utils.hashing import stable_id
from app.utils.text import chunk_text

#: Headings are markdown-ish, which is what the extraction layer produces.
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 150


def _heading_for(text: str, position: int) -> str | None:
    """The heading in force at ``position`` — the last one before it.

    Scanned by position rather than tracked during the split, so a chunk built
    from a carried-over overlap still reports the heading its text sits under.
    """
    current: str | None = None
    for match in HEADING_PATTERN.finditer(text):
        if match.start() > position:
            break
        current = match.group(2).strip()
    return current


def _offset_of(document_content: str, piece: str, search_from: int) -> int:
    """Where ``piece`` starts in the document, for heading lookup.

    Falls back to the running offset when the text was shortened or whitespace-
    collapsed on the way through, so a heading lookup degrades to "no heading"
    rather than to a wrong one.
    """
    index = document_content.find(piece, search_from)
    return index if index >= 0 else search_from


def chunk_document(
    document: Document,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split a document into chunks that each carry their own provenance."""
    content = document.content or ""
    if not content.strip():
        return []

    pieces = chunk_text(content, max_chars=max_chars, overlap=overlap)
    chunks: list[Chunk] = []
    cursor = 0
    for ordinal, piece in enumerate(pieces):
        position = _offset_of(content, piece, cursor)
        cursor = max(cursor, position + max(1, len(piece) - overlap))
        chunks.append(
            Chunk(
                id=stable_id("chunk", document.id, ordinal),
                document_id=document.id,
                run_id=document.run_id,
                source_id=document.source_id,
                url=document.canonical_url or document.url,
                text=piece,
                ordinal=ordinal,
                company=document.company,
                competitor=document.competitor,
                page_type=document.page_type,
                title=document.title,
                heading=_heading_for(content, position),
                # A copy, not a reference: a chunk outliving its document's
                # metadata dict must not change meaning underneath its index.
                metadata=dict(document.metadata),
            )
        )
    return chunks


def chunk_documents(
    documents: list[Document],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(document, max_chars=max_chars, overlap=overlap)
        )
    return chunks
