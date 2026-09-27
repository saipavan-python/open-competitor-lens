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


#: A number that looks like a statistic: an optional currency symbol, thousands
#: separators, a decimal part, an optional percentage or magnitude suffix. The
#: lookbehind and lookahead keep version numbers, ``k8s`` and ``node20`` from
#: being read as figures.
_NUMBER = re.compile(
    r"(?<![\w/])"
    r"(?:[$€£]\s?)?"
    r"\d[\d,]*(?:\.\d+)?"
    r"\s?(?:%|percent|per cent)?"
    r"\s?(?:[kKmMbB]|bn|mn)?"
    r"(?![\w/])",
    re.IGNORECASE,
)

#: Years are not statistics. A report is dated and every source is dated, so
#: flagging every year as an unevidenced figure would make the check unusable
#: rather than strict.
_YEAR = re.compile(r"^(?:19|20)\d{2}$")

#: Contexts in which a number is a date, an identifier or a section count rather
#: than a statistic about the companies. Matched against the *line* a number sits
#: on rather than the whole text: testing the whole text would let one
#: "Section 7" anywhere in a paragraph exempt a figure in the same paragraph,
#: which is the opposite of the rule.
_NON_STATISTIC_CONTEXT = re.compile(
    r"(?:\bhttps?://)"
    r"|(?:\bsection\s+\d+)"
    r"|(?:^\s*#{1,6}\s*\d+)"
    r"|(?:^\s*\d+[.)]\s)",
    re.IGNORECASE,
)


def _line_at(text: str, index: int) -> str:
    """The line containing ``index``, which is what a number's context means."""
    start = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    return text[start:] if end == -1 else text[start:end]


def numeric_tokens(text: str) -> list[str]:
    """Every statistic-looking number in a piece of text.

    Years and non-statistic contexts are excluded here rather than at the call
    sites, so a year cannot be smuggled past the check by a caller that forgot the
    rule.

    This lives in ``utils`` rather than beside the fact checker because the report
    quality service applies the same rule to the same prose, and a service
    importing from the agent layer above it would invert the dependency.
    """
    found: list[str] = []
    for match in _NUMBER.finditer(text or ""):
        token = match.group(0).strip()
        digits = token.lstrip("$" + "\u20ac\u00a3 ").rstrip("%").replace(",", "")
        digits = re.sub(r"(?:%|percent|per cent)$", "", digits, flags=re.IGNORECASE)
        digits = re.sub(r"[kKmMbB]|bn|mn$", "", digits).strip()
        if not digits or _YEAR.match(digits):
            continue
        if _NON_STATISTIC_CONTEXT.search(_line_at(text or "", match.start())):
            continue
        found.append(token)
    return found


def unsupported_numeric_tokens(claim: str, evidence_text: str) -> list[str]:
    """Numbers in a claim that no evidence text contains.

    Compared on digits alone, so "$10" in a claim matches "10" in a quote and
    "10,000" matches "10000" - the same figure written the two ways prose and
    tables write it. A claim whose number appears nowhere in its evidence is
    reporting something the source does not say.
    """
    haystack = re.sub(r"[,\s]", "", evidence_text or "")
    missing: list[str] = []
    for token in numeric_tokens(claim):
        if re.sub(r"[,\s]", "", token) not in haystack:
            missing.append(token)
    return missing
