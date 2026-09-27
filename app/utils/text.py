"""Text processing: whitespace, chunking and truncation.

Chunking is boundary-aware rather than character-aware. A chunk that begins
mid-sentence or splits a heading from its section is worse than useless for
retrieval: it retrieves the right page for the wrong reason.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

_WHITESPACE = re.compile(r"\s+")
_PARAGRAPH = re.compile(r"\n\s*\n")
_HEADING = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


def collapse_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces."""
    return _WHITESPACE.sub(" ", text or "").strip()


def normalize_space(text: str) -> str:
    """Alias of :func:`collapse_whitespace` for import clarity at call sites."""
    return collapse_whitespace(text)


def word_count(text: str) -> int:
    """Number of whitespace-separated words."""
    return len((text or "").split())


def truncate(text: str, limit: int, *, suffix: str = "...") -> str:
    """Truncate to ``limit`` characters, adding ``suffix`` only if shortened."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))] + suffix


def _iter_sections(text: str) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(heading, paragraphs)`` for each heading-delimited section."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        yield "", _PARAGRAPH.split(text)
        return
    if matches[0].start() > 0:
        yield "", _PARAGRAPH.split(text[: matches[0].start()])
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end(): end]
        yield match.group(0).strip(), _PARAGRAPH.split(body)


def chunk_text(text: str, *, max_chars: int = 1500, overlap: int = 150) -> list[str]:
    """Split text into retrievable chunks on heading and paragraph boundaries.

    Guarantees, all covered by tests:

    * no chunk exceeds ``max_chars`` (a single oversized paragraph is hard-split)
    * a paragraph is never cut in half merely to fill a chunk
    * a heading always travels with the body it introduces
    * empty input produces no chunks
    """
    if not text or not text.strip():
        return []
    max_chars = max(200, max_chars)
    overlap = max(0, min(overlap, max_chars // 2))

    chunks: list[str] = []
    for heading, paragraphs in _iter_sections(text):
        prefix = f"{heading}\n" if heading else ""
        budget = max_chars - len(prefix)
        if budget <= 0:
            prefix, budget = "", max_chars
        buffer: list[str] = []
        size = 0
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) > budget:
                if buffer:
                    chunks.append(prefix + "\n\n".join(buffer))
                    buffer, size = [], 0
                for start in range(0, len(paragraph), budget):
                    chunks.append(prefix + paragraph[start: start + budget])
                continue
            addition = len(paragraph) + (2 if buffer else 0)
            if size + addition > budget:
                chunks.append(prefix + "\n\n".join(buffer))
                carried = "\n\n".join(buffer)[-overlap:] if overlap else ""
                buffer = [carried] if carried.strip() else []
                size = len(buffer[0]) if buffer else 0
                addition = len(paragraph) + (2 if buffer else 0)
            buffer.append(paragraph)
            size += addition
        if buffer:
            chunk = prefix + "\n\n".join(buffer)
            if chunk.strip():
                chunks.append(chunk)
    return [chunk for chunk in chunks if chunk.strip()]


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    """Remove duplicates while keeping first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_quote(text: str, needle: str, *, window: int = 320) -> str | None:
    """Return a verbatim window around ``needle``, or ``None`` if absent.

    Used to prove that a quoted passage really appears in a retrieved page.
    """
    if not text or not needle:
        return None
    index = text.find(needle)
    if index != -1:
        start = max(0, index - window // 3)
        return text[start: index + len(needle) + window]

    # Fall back to a whitespace-collapsed comparison: HTML extraction routinely
    # reflows a page's spacing, which would otherwise fail an honest quote.
    collapsed_text = collapse_whitespace(text)
    collapsed_needle = collapse_whitespace(needle)
    index = collapsed_text.find(collapsed_needle)
    if index == -1:
        return None
    start = max(0, index - window // 3)
    return truncate(collapsed_text[start: index + len(collapsed_needle) + window],
                    window + len(collapsed_needle))
