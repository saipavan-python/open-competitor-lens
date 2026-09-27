"""The evidence gate: nothing enters a report without a retrieved source.

Three things are checked on every attach, and each exists because a language
model will cheerfully do the thing it prevents:

1. **The source was actually fetched by this run.** A model asked to cite its
   findings will produce a plausible URL. If the run never retrieved that URL,
   the citation is decoration.
2. **The quote is really in the retrieved text.** Compared against the text this
   service stored, not against a copy the caller passed in — otherwise a caller
   can supply the "page" the quote came from and the check verifies nothing.
3. **Confidence does not outrank the claim type or the source's authority.** An
   inference is not an observation however fluent it sounds, and a blog post is
   not a primary source however confident the model is.

Conflicts are recorded, never resolved silently. Two sources disagreeing about a
price is a finding the report should carry, and quietly keeping whichever was
retrieved first would be a fabrication of consensus.
"""

from __future__ import annotations

from app.core.constants import ClaimType, Confidence, RetrievalStatus
from app.core.exceptions import EvidenceNotSupportedError
from app.core.logging import get_logger
from app.schemas.evidence import Evidence, EvidenceBundle, Source
from app.utils.hashing import stable_id
from app.utils.text import collapse_whitespace

log = get_logger(__name__)

#: Characters that differ between a source page and a model's rendering of it.
#: Curly quotes and non-breaking spaces are the usual culprits, and a quote that
#: fails on those alone is not a fabricated quote.
_NORMALISE = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u2026": "...",
    "\u200b": "", "\ufeff": "",
})


def _normalise(text: str) -> str:
    return collapse_whitespace((text or "").translate(_NORMALISE).lower())


def quote_appears_in(quote: str, text: str) -> bool:
    """Whether ``quote`` really occurs in ``text``.

    Tolerant of the differences extraction and generation legitimately introduce
    — collapsed whitespace, curly punctuation, case — and intolerant of
    everything else. A quote that does not appear is refused even if it is
    semantically equivalent, because "close enough" is how a paraphrase becomes
    a quotation.
    """
    if not quote or not text:
        return False
    if quote.strip() in text:
        return True
    return _normalise(quote) in _normalise(text)


class EvidenceService:
    """Collects the evidence for one run and enforces the traceability rules."""

    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}
        self._evidence: list[Evidence] = []
        self._by_key: dict[str, list[Evidence]] = {}
        self._rejected: list[str] = []

    def __len__(self) -> int:
        """The number of sources this run fetched.

        Sources, not evidence: the size of a run is bounded by what was
        retrieved, and a caller asking ``len(service)`` is asking how much
        ground the run actually covered. Evidence is counted with
        :meth:`all_evidence`.
        """
        return len(self._sources)

    # -- run bookkeeping -------------------------------------------------

    def register_fetched(self, source: Source) -> Source:
        """Record a source this run retrieved. Idempotent by id."""
        self._sources[source.id] = source
        return source

    def register_fetched_many(self, sources: list[Source]) -> list[Source]:
        return [self.register_fetched(source) for source in sources]

    def fetched(self) -> list[Source]:
        return list(self._sources.values())

    def all_evidence(self) -> list[Evidence]:
        return list(self._evidence)

    def reset(self) -> None:
        self._sources.clear()
        self._evidence.clear()
        self._by_key.clear()
        self._rejected.clear()

    # -- the gate --------------------------------------------------------

    def attach(
        self,
        claim: str,
        *,
        source: Source,
        quote: str,
        claim_type: ClaimType,
        confidence: Confidence,
        claim_key: str | None = None,
        competitor: str | None = None,
        category: str | None = None,
    ) -> Evidence:
        """Verify and record one source-backed claim, or raise.

        Returns the stored :class:`Evidence`. Raises
        :class:`EvidenceNotSupportedError` if the source was not fetched, did
        not retrieve successfully, does not contain the quote, or cannot support
        the requested confidence.
        """
        claim = (claim or "").strip()
        if not claim:
            raise EvidenceNotSupportedError("a claim may not be empty", claim=claim)

        stored = self._sources.get(source.id)
        if stored is None:
            raise EvidenceNotSupportedError(
                f"Source '{source.id}' was not retrieved by this run, so nothing "
                f"can be cited from it. A citation must point at a page this "
                f"analysis actually fetched.",
                claim=claim, source_id=source.id,
            )

        if stored.url != source.url or stored.text != source.text:
            raise EvidenceNotSupportedError(
                f"The source supplied for '{source.id}' is not the page this run "
                f"retrieved: its text or URL differs from the stored copy. "
                f"Evidence is verified against what was fetched, so a caller "
                f"cannot substitute its own version of the page.",
                claim=claim, source_id=source.id,
            )

        if stored.retrieval_status is not RetrievalStatus.OK:
            raise EvidenceNotSupportedError(
                f"Source '{stored.id}' has retrieval status "
                f"'{stored.retrieval_status.value}', so it supports no claim.",
                claim=claim, source_id=stored.id,
            )

        if not stored.text:
            raise EvidenceNotSupportedError(
                f"Source '{stored.id}' stored no text, so no quote can be verified "
                f"against it.",
                claim=claim, source_id=stored.id,
            )

        if not quote_appears_in(quote, stored.text):
            raise EvidenceNotSupportedError(
                f"The supplied quote does not appear in the text retrieved from "
                f"'{stored.url}'. A quote that is not in the source is not "
                f"evidence.",
                claim=claim, source_id=stored.id,
            )

        self._check_confidence(claim, stored, claim_type, confidence)

        evidence = Evidence(
            id=stable_id("ev", claim_key or claim, stored.id, quote[:120]),
            claim=claim,
            source_id=stored.id,
            source_url=stored.url,
            evidence_text=quote.strip(),
            claim_type=claim_type,
            confidence=confidence,
            claim_key=claim_key,
            company=stored.company,
            competitor=competitor or stored.competitor,
            category=category,
            source_type=stored.source_type,
            published_at=stored.published_at,
        )
        self._record(evidence)
        return evidence

    def _check_confidence(
        self, claim: str, source: Source, claim_type: ClaimType, confidence: Confidence
    ) -> None:
        if claim_type is ClaimType.INFERRED and confidence is Confidence.HIGH:
            raise EvidenceNotSupportedError(
                f"An inference cannot carry high confidence: '{claim}' is the "
                f"model's reasoning, not something the source states.",
                claim=claim, source_id=source.id,
            )
        if claim_type is ClaimType.UNKNOWN and confidence is Confidence.HIGH:
            raise EvidenceNotSupportedError(
                f"An unknown claim cannot carry high confidence: '{claim}' was "
                f"not established by any source.",
                claim=claim, source_id=source.id,
            )
        if confidence is Confidence.HIGH and not source.supports(confidence):
            raise EvidenceNotSupportedError(
                f"Source '{source.id}' is a {source.source_type.value} and is not "
                f"a primary source, so it cannot support high confidence for "
                f"'{claim}'.",
                claim=claim, source_id=source.id,
            )

    # -- conflicts and bundles -------------------------------------------

    def _record(self, evidence: Evidence) -> None:
        if evidence.claim_key:
            peers = self._by_key.setdefault(evidence.claim_key, [])
            conflicts = [peer for peer in peers if _disagrees(peer, evidence)]
            if conflicts:
                evidence.conflicts_with = sorted(item.id for item in conflicts)
                for peer in conflicts:
                    peer.conflicts_with = sorted(
                        set(peer.conflicts_with) | {evidence.id}
                    )
            peers.append(evidence)
        self._evidence.append(evidence)

    def bundle_for(self, claim_key: str) -> EvidenceBundle | None:
        """Group the evidence for one ``claim_key``, or ``None`` if there is none."""
        items = self._by_key.get(claim_key)
        if not items:
            return None
        best = max(
            (item.confidence for item in items),
            key=lambda level: _CONFIDENCE_ORDER.index(level),
        )
        return EvidenceBundle(
            claim=items[0].claim,
            evidence=list(items),
            conflicts=any(item.conflicts_with for item in items),
            best_confidence=best,
        )

    def conflicts(self) -> dict[str, list[Evidence]]:
        """Every claim key whose evidence disagrees, for the report to state."""
        return {
            key: items
            for key, items in self._by_key.items()
            if any(item.conflicts_with for item in items)
        }

    def reject(self, claim: str, reason: str) -> str:
        """Record a claim the run wanted to make but could not support.

        Rejections are surfaced rather than dropped. A report that states
        "market share is unknown" is a different and more useful artefact than
        one that never mentions market share, because a reader can tell the
        difference between a question nobody thought to ask and one that was
        asked and went unanswered. That distinction is the whole reason this
        exists rather than a bare ``except: pass``.

        Returns the normalised claim, so the caller can report the same string
        it recorded here as the run's unknown sentinel.
        """
        cleaned = " ".join((claim or "").split())
        if not cleaned:
            raise ValueError("a rejected claim may not be empty")
        if cleaned not in self._rejected:
            self._rejected.append(cleaned)
        log.info(
            "claim rejected as unsupported",
            extra={"reason": " ".join(reason.split())[:300]},
        )
        return cleaned

    def unsupported_claims(self) -> list[str]:
        """Claims the run wanted to make but could not support.

        Populated by :meth:`reject`. These are reported as the spec's explicit
        unknown sentinel rather than dropped, so that a reader can tell the
        difference between "we found nothing" and "we found nothing to find".
        """
        return list(self._rejected)


def _disagrees(left: Evidence, right: Evidence) -> bool:
    """Whether two evidences about the same key state different things.

    Confidence is deliberately excluded. A second source agreeing at a higher
    confidence strengthens a claim rather than contradicting it, and counting
    that as a conflict would file a consensus as a dispute -- the report would
    warn readers about a conflict that does not exist.
    """
    return _substance(left) != _substance(right)


def _substance(evidence: Evidence) -> str:
    """What an evidence says, reduced to a comparable form.

    The quotation is compared as well as the claim, because two sources can
    carry identical claim wording and disagree in the figures backing it. The
    disagreement most worth surfacing usually lives in the quote, not the
    summary of it.
    """
    return (
        f"{_normalise(evidence.claim)}"
        f"|{_normalise(evidence.evidence_text or '')}"
    )


_CONFIDENCE_ORDER = [Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH]
