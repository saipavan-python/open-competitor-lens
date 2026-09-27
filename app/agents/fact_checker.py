"""The fact checker: every claim in the report, against the evidence.

The pipeline's other checks run *before* a claim exists. This one runs after,
reading the assembled report as a reader will meet it, and it is the last
opportunity to notice that a sentence is not standing on anything.

A model asked to verify its own output verifies it generously — the claim reads
as true because the model wrote it moments ago, and the evidence it is handed
was chosen to support it. So this agent uses no model at all. It applies stated
rules to stated evidence, and every verdict it can reach is one a reader could
have reached by opening the same links:

``supported``
    Evidence backs the claim, at a strength matching its confidence.
``overstatement``
    The claim says more than its evidence does. A superlative on the strength of
    one page, or a number that does not appear in the quote.
``unsupported``
    Nothing backs it, or it cites evidence the run never attached.
``contradictory``
    The evidence disagrees with itself.
``stale``
    The evidence is old enough that a pricing or headcount claim built on it
    should not be presented as current.

The order these are checked in matters, and it is deliberate. Contradiction is
tested before support so that a conflict can never read as agreement, and
staleness before support for the same reason. A rule checked late is a rule that
loses.

**Confidence only ever goes down.** A checker's job is to subtract confidence,
and any rule that could raise it would let a run launder a low-confidence claim
into a high-confidence one by having it re-examined. The most this module can do
to a claim's confidence is hold it or lower it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from app.core.constants import NON_PRIMARY_SOURCE_TYPES, ClaimVerdict, Confidence, SourceType
from app.core.logging import get_logger
from app.schemas.evidence import Evidence
from app.schemas.finding import ClaimVerdictRecord
from app.utils.text import numeric_tokens, unsupported_numeric_tokens

log = get_logger(__name__)

#: Comparative language that no single page can support. Each entry is matched
#: case-insensitively as a phrase; the list is about claims that assert a ranking,
#: a size or a superiority, because those are the ones a vendor's own marketing
#: page is most likely to make and least likely to have earned.
SUPERLATIVE_PATTERNS: tuple[str, ...] = (
    r"market leader",
    r"industry leader",
    r"\bleader\b",
    r"\bthe best\b",
    r"\bbest[- ]in[- ]class\b",
    r"number one",
    r"\b#1\b",
    r"\bmost (?:popular|used|trusted|recognised|recognized|advanced)\b",
    r"\b(?:largest|fastest|cheapest|strongest)\b",
    r"\bwidely used\b",
    r"\bindustry[- ]standard\b",
    r"\bdominant\b",
    r"\bunrivalled\b|\bunrivaled\b",
)

_COMPILED_SUPERLATIVES = tuple(re.compile(p, re.IGNORECASE) for p in SUPERLATIVE_PATTERNS)

#: A number as prose writes it: currency, thousands separators, decimals,
#: percentages and magnitude suffixes. The trailing boundary rejects a digit run
#: that is part of a longer identifier, so a product code or a hash is not read as
#: a figure — but a magnitude letter is part of the figure, so "4M" is one token.
#: Evidence older than this about a commercial fact is stale rather than current.
DEFAULT_MAX_AGE_DAYS = 365

#: Distinct primary sources needed to sustain a high-confidence claim.
HIGH_CONFIDENCE_SOURCES = 2


def superlatives_in(claim: str) -> list[str]:
    """Comparative phrases in a claim, longest match first, in order of position.

    Longer patterns are tried first and their spans are then removed before the
    shorter ones run, so "market leader" is reported as one phrase rather than as
    "market leader" *and* "leader". Two entries for one phrase would make a
    reason read as though the claim had made two separate overstatements.
    """
    spans: list[tuple[int, int]] = []
    found: list[str] = []
    for pattern in sorted(_COMPILED_SUPERLATIVES, key=lambda p: -len(p.pattern)):
        for match in pattern.finditer(claim):
            start, end = match.span()
            if any(start < used_end and end > used_start for used_start, used_end in spans):
                continue
            spans.append((start, end))
            text = match.group(0).strip()
            if text and text not in found:
                found.append(text)
    return sorted(found, key=lambda text: claim.find(text))








class FactCheckerAgent:
    """Applies the verdict rules to each claim, deterministically."""

    name = "fact_checker"
    goal = "Check every report claim against the evidence the run gathered."

    def __init__(self, *, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> None:
        self._max_age = timedelta(days=max_age_days)

    def verify(
        self,
        claims: Iterable[Mapping[str, Any]],
        evidence_index: Mapping[str, Sequence[Evidence]],
    ) -> list[ClaimVerdictRecord]:
        """A ruling for every claim, in the order given.

        ``evidence_index`` maps a claim id to the evidence gathered for it. A claim
        naming evidence ids the index does not hold is unsupported, which is
        different from a claim with no evidence at all and is stated as such: one
        is a broken reference, the other is a gap in the research.
        """
        return [self._verify_one(claim, evidence_index) for claim in claims]

    # -- internals --------------------------------------------------------

    def _verify_one(
        self,
        claim: Mapping[str, Any],
        evidence_index: Mapping[str, Sequence[Evidence]],
    ) -> ClaimVerdictRecord:
        text = str(claim.get("claim", "")).strip()
        claim_id = str(claim.get("claim_id") or claim.get("id") or _fallback_id(text))
        before = _as_confidence(claim.get("confidence"))
        named = [str(item) for item in claim.get("evidence_ids", []) or []]
        gathered = list(evidence_index.get(claim_id, []) or [])
        usable = [item for item in gathered if item.id in named] if named else gathered

        verdict, reason, supporting, contradicting = self._rule(
            text, usable, gathered, named
        )
        after = _cap(_as_confidence(self._supported_confidence(usable, before)), before)

        if verdict in (ClaimVerdict.UNSUPPORTED, ClaimVerdict.OVERSTATEMENT):
            log.info(
                "claim did not survive the fact check",
                extra={
                    "claim_id": claim_id,
                    "verdict": verdict.value,
                    "evidence": len(usable),
                },
            )
        return ClaimVerdictRecord(
            claim_id=claim_id,
            claim=text,
            verdict=verdict,
            reason=reason,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            confidence_before=before,
            confidence_after=after,
        )

    def _rule(
        self,
        claim: str,
        usable: Sequence[Evidence],
        gathered: Sequence[Evidence],
        named: Sequence[str],
    ) -> tuple[ClaimVerdict, str, list[str], list[str]]:
        """The verdict rules, in the order they are allowed to fire.

        Contradiction and staleness are tested before support, so neither can be
        read as agreement; unsupported is tested before overstatement, because a
        claim with no evidence is not "saying too much", it is saying nothing.
        """
        held_ids = {item.id for item in gathered}
        dangling = [ref for ref in named if ref not in held_ids]
        if dangling:
            return (
                ClaimVerdict.UNSUPPORTED,
                f"cites evidence the run never attached: {', '.join(sorted(dangling))}",
                [],
                [],
            )

        if not usable:
            return (
                ClaimVerdict.UNSUPPORTED,
                "no evidence was retrieved for this claim",
                [],
                [],
            )

        contradicting = sorted(
            {ref for item in usable for ref in item.conflicts_with}
        )
        if contradicting:
            return (
                ClaimVerdict.CONTRADICTORY,
                f"the evidence disagrees with itself: {', '.join(contradicting)}",
                [item.id for item in usable],
                contradicting,
            )

        stale = [item.id for item in usable if self._is_stale(item)]
        if stale:
            return (
                ClaimVerdict.STALE,
                f"rests on evidence older than {self._max_age.days} days: "
                f"{', '.join(stale)}",
                [item.id for item in usable],
                [],
            )

        superlatives = superlatives_in(claim)
        if superlatives:
            return (
                ClaimVerdict.OVERSTATEMENT,
                "asserts a ranking or superiority its evidence does not establish: "
                f"{', '.join(superlatives)}",
                [item.id for item in usable],
                [],
            )

        unsupported_numbers = self._unsupported_numbers(claim, usable)
        if unsupported_numbers:
            return (
                ClaimVerdict.OVERSTATEMENT,
                f"cites a figure absent from its evidence: {', '.join(unsupported_numbers)}",
                [item.id for item in usable],
                [],
            )

        return (
            ClaimVerdict.SUPPORTED,
            f"backed by {len({item.source_id for item in usable})} source(s) this run "
            f"retrieved",
            [item.id for item in usable],
            [],
        )

    def _unsupported_numbers(self, claim: str, usable: Sequence[Evidence]) -> list[str]:
        """Figures in the claim that none of its evidence text contains."""
        quotes = " ".join(item.evidence_text or "" for item in usable)
        if not quotes.strip():
            return []
        return unsupported_numeric_tokens(claim, quotes)

    def _is_stale(self, item: Evidence) -> bool:
        """Whether an evidence record is too old to speak in the present tense.

        Only dated evidence can go stale. An evidence record with no publication
        date is recent as far as this system knows, and treating undated evidence
        as old would make every claim in a report stale by default.
        """
        if item.published_at is None:
            return False
        reference = datetime.now(item.published_at.tzinfo)
        return reference - item.published_at > self._max_age

    def _supported_confidence(
        self, usable: Sequence[Evidence], before: Confidence
    ) -> Confidence:
        """The most confidence this evidence could support, if it is all primary."""
        if not usable:
            return Confidence.LOW
        sources = {item.source_id for item in usable}
        primary = {
            item.source_id
            for item in usable
            if item.source_type not in NON_PRIMARY_SOURCE_TYPES
        }
        if len(primary) >= HIGH_CONFIDENCE_SOURCES:
            return Confidence.HIGH
        if len(sources) >= 1:
            return Confidence.MEDIUM
        return Confidence.LOW


def _as_confidence(value: Any) -> Confidence:
    """A confidence from a model-supplied value, defaulting to low.

    Low rather than high, because a claim that arrives without saying how sure it
    is has told us nothing, and reading silence as confidence is the error this
    whole system is arranged against.
    """
    if isinstance(value, Confidence):
        return value
    try:
        return Confidence(str(value))
    except ValueError:
        return Confidence.LOW


def _cap(value: Confidence, ceiling: Confidence) -> Confidence:
    """Clamp a confidence to its ceiling — this module only ever lowers."""
    order = [Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH]
    return value if order.index(value) <= order.index(ceiling) else ceiling


def _fallback_id(text: str) -> str:
    """A stable id for a claim that arrived without one."""
    return f"claim-{abs(hash(text)) % 10_000_000:07d}"


#: Re-exported so callers can reason about source tiers without a second import.
__all__ = [
    "DEFAULT_MAX_AGE_DAYS",
    "FactCheckerAgent",
    "HIGH_CONFIDENCE_SOURCES",
    "SourceType",
    "numeric_tokens",
    "superlatives_in",
    "unsupported_numeric_tokens",
]
