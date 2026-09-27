"""Content extraction, page identity and de-duplication.

The rules here are the ones the crawler's correctness rests on, and they are
deliberately conservative in the direction of *keeping* sources:

- Identity is the **canonical URL**. Two URLs with identical bytes are two
  sources, because they are two pages and a report may legitimately cite both.
- The content key is ``sha256(canonical_url + content_hash)``. Including the URL
  in the hash is what stops the byte-identical legal footer that every SaaS
  website ships from collapsing two competitors into one.
- The scope is **per company**, so a shared template cannot merge competitors.
- A failed, missing or blocked retrieval records nothing. It must not shadow a
  later successful retry of the same URL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.schemas.provider import FetchResult
from app.utils.hashing import sha256_text
from app.utils.urls import canonicalize_url

log = get_logger(__name__)

#: Only a successful retrieval can become a source.
_RECORDED_STATUSES = frozenset({"ok"})


@dataclass(frozen=True)
class DedupDecision:
    """What the deduplicator decided about one page, and why."""

    url: str
    key: str
    content_key: str
    is_new: bool
    is_duplicate: bool
    changed: bool
    recorded: bool

    @property
    def should_keep(self) -> bool:
        """Whether this page should appear in the result set."""
        return self.is_new and self.recorded


@dataclass
class _Entry:
    key: str
    content_key: str
    url: str


class Deduplicator:
    """Tracks which pages have already been seen, within one company scope."""

    def __init__(self, *, scope: str = "default") -> None:
        self.scope = scope
        self._by_key: dict[str, _Entry] = {}
        self._content_keys: set[str] = set()

    def __len__(self) -> int:
        return len(self._by_key)

    @property
    def count(self) -> int:
        return len(self._by_key)

    @staticmethod
    def key_for(url: str) -> str:
        """A page's identity: its canonical URL, or the raw URL if unparseable."""
        return canonicalize_url(url) or url

    @staticmethod
    def content_key_for(key: str, result: FetchResult) -> str:
        """Identity of a page *plus* its content, for change detection."""
        digest = result.content_hash or sha256_text(result.text)
        return sha256_text(f"{key}:{digest}")

    def seen(self, url: str) -> bool:
        """Whether this URL has already been recorded, without recording it."""
        return self.key_for(url) in self._by_key

    def seen_content(self, key: str, result: FetchResult) -> bool:
        return self.content_key_for(key, result) in self._content_keys

    def add(self, result: FetchResult) -> DedupDecision:
        """Record a page and report what happened."""
        key = self.key_for(result.final_url or result.url)
        content_key = self.content_key_for(key, result)
        existing = self._by_key.get(key)

        if result.retrieval_status not in _RECORDED_STATUSES:
            # Nothing to record, and nothing that should block a later retry.
            return DedupDecision(
                url=result.url, key=key, content_key=content_key, is_new=False,
                is_duplicate=False, changed=False, recorded=False,
            )

        if existing is not None:
            return DedupDecision(
                url=result.url, key=key, content_key=content_key, is_new=False,
                is_duplicate=True,
                changed=existing.content_key != content_key,
                recorded=True,
            )

        self._by_key[key] = _Entry(key=key, content_key=content_key, url=result.url)
        self._content_keys.add(content_key)
        return DedupDecision(
            url=result.url, key=key, content_key=content_key, is_new=True,
            is_duplicate=False, changed=False, recorded=True,
        )

    def deduplicate(
        self, results: list[FetchResult], *, return_decisions: bool = False
    ) -> list[FetchResult] | tuple[list[FetchResult], list[DedupDecision]]:
        """Filter a batch down to first occurrences, preserving order."""
        kept: list[FetchResult] = []
        decisions: list[DedupDecision] = []
        for result in results:
            decision = self.add(result)
            decisions.append(decision)
            if decision.should_keep:
                kept.append(result)
        if return_decisions:
            return kept, decisions
        return kept

    def reset(self) -> None:
        self._by_key.clear()
        self._content_keys.clear()


@dataclass
class ExtractionResult:
    """Structured output from one page, plus where it came from."""

    source_id: str | None
    url: str
    data: dict = field(default_factory=dict)
    quote: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.data)


class ExtractionService:
    """Turns retrieved text into schema-shaped data.

    Deliberately does not invent values. When a field cannot be filled from the
    page it is left absent, and the report carries the spec's explicit unknown
    sentinel rather than a plausible guess — the whole point of the evidence
    system is that a missing fact is visible.
    """

    def __init__(self, llm: object | None = None) -> None:
        self.llm = llm

    async def extract(self, result: FetchResult, schema: type) -> ExtractionResult:
        if not result.ok:
            return ExtractionResult(
                source_id=None, url=result.final_url or result.url,
                error=result.error or f"retrieval status {result.retrieval_status}",
            )
        if self.llm is None:
            return ExtractionResult(
                source_id=None, url=result.final_url or result.url,
                data={"text": result.text},
            )
        response = await self.llm.generate(  # type: ignore[attr-defined]
            self._prompt(result, schema), schema=schema
        )
        return ExtractionResult(
            source_id=None, url=result.final_url or result.url,
            data=getattr(response, "parsed", None) or {},
        )

    def _prompt(self, result: FetchResult, schema: type) -> str:
        return (
            "Extract the requested fields from the page below. Use only text "
            "present in the page. If a field is not stated, omit it.\n\n"
            f"PAGE URL: {result.final_url or result.url}\n"
            f"PAGE TEXT:\n{result.text}"
        )
