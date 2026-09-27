"""Attaching evidence to claims, and refusing the claims that will not take it.

Task 7 built :class:`~app.services.evidence_service.EvidenceService`, which
verifies that a quoted passage really appears in the page the run retrieved. This
agent is the caller that turns an analyst's output into calls on it, and its whole
purpose is to make sure a claim which fails verification is *reported* as failing
rather than quietly dropped.

That distinction is the reason this is an agent and not a loop in the graph. A
silent ``except`` here would turn three unverified claims into a clean-looking
report with three fewer sentences, and nothing downstream could tell. Instead every
rejection is counted, named, and carried in the result's warnings, so a reader can
be told "three claims could not be verified" — which is a finding about the
research — rather than being handed a shorter report with no explanation.

The other rule is about *who* decides. Verification lives in the service and is
not overridable here. An agent cannot pass a claim through because the model was
confident, and there is no ``force`` parameter, because the first caller to need
one would be the caller who decides the whole rule was inconvenient.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.agents.base import Agent, AgentContext, AgentResult, AgentStatus
from app.core.constants import ClaimType, Confidence
from app.core.exceptions import EvidenceNotSupportedError
from app.core.logging import get_logger
from app.schemas.evidence import Evidence, Source
from app.services.evidence_service import EvidenceService

log = get_logger(__name__)


class EvidenceAgent(Agent):
    """Registers fetched sources and attaches verified evidence to claims."""

    name = "evidence"
    goal = "Turn claims and quotes into evidence records, refusing any that do not verify."

    def __init__(self, *, evidence: EvidenceService | None = None) -> None:
        self._evidence = evidence or EvidenceService()

    @property
    def service(self) -> EvidenceService:
        return self._evidence

    async def run(self, ctx: AgentContext) -> AgentResult:
        """Attach every verifiable claim, and report every one that would not attach.

        A run where nothing attached is a failure: the report has no evidence at
        all, and that is not the same as a report with a few gaps in it.
        """
        sources = self._register(ctx.extras.get("sources", []) or [])
        claims = list(ctx.extras.get("claims", []) or [])
        if not claims:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.PARTIAL,
                warnings=["no claims were supplied, so no evidence was attached"],
            )

        attached: list[Evidence] = []
        rejected: list[str] = []
        for claim in claims:
            record = self._attach(claim, sources)
            if isinstance(record, Evidence):
                attached.append(record)
            else:
                rejected.append(str(record))

        if not attached:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.FAILED,
                errors=[
                    f"none of {len(claims)} claim(s) could be verified against the pages "
                    f"this run retrieved"
                ],
                warnings=[f"rejected: {reason}" for reason in rejected],
            )

        result = AgentResult(
            agent=self.name,
            status=AgentStatus.PARTIAL if rejected else AgentStatus.OK,
            payload=attached,
            retrieved_source_ids=[source.id for source in sources],
            evidence_ids=[item.id for item in attached],
        )
        if rejected:
            # Accumulated rather than raised: three refused claims are a
            # systematic problem with one analyst's output, and reporting the
            # first alone would hide the other two.
            result.warnings.append(
                f"{len(rejected)} claim(s) could not be verified: {'; '.join(rejected)}"
            )
        return result

    # -- internals --------------------------------------------------------

    def _register(self, sources: Iterable[Source | Mapping[str, Any]]) -> list[Source]:
        """Register the pages the run retrieved, so they can be cited.

        A source that is not already a :class:`Source` is refused rather than
        coerced: the fields that make a source citable (its text, its type) cannot
        be inferred, and a source with no text can never verify a quote.
        """
        usable = [item for item in sources if isinstance(item, Source)]
        skipped = len(list(sources)) - len(usable)
        if skipped:
            log.warning(
                "ignored sources that were not fully specified",
                extra={"count": skipped},
            )
        return self._evidence.register_fetched_many(usable) if usable else []

    def _attach(self, claim: Mapping[str, Any], sources: Sequence[Source]) -> Evidence | str:
        """Attach one claim, or return why it could not be attached.

        Returns the reason rather than raising, because a refusal is ordinary
        here — most refusals are a quote the model paraphrased — and an exception
        would abandon the claims that would have verified.
        """
        text = str(claim.get("claim", "")).strip()
        if not text:
            return "empty claim"
        source_id = str(claim.get("source_id", "")).strip()
        source = next((item for item in sources if item.id == source_id), None)
        if source is None:
            return f"{text!r} cites source {source_id!r}, which this run did not retrieve"
        quote = str(claim.get("quote", "")).strip()
        if not quote:
            return f"{text!r} has no quote to verify"
        try:
            return self._evidence.attach(
                text,
                source=source,
                quote=quote,
                claim_type=_claim_type(claim.get("claim_type")),
                confidence=_confidence(claim.get("confidence")),
                claim_key=claim.get("claim_key") or text,
                competitor=claim.get("competitor"),
                category=claim.get("category"),
            )
        except EvidenceNotSupportedError as exc:
            # The service signals *every* refusal with this type — including the
            # two cases this system exists to catch, a quote that is not in the
            # page and a source the run never fetched. Catching ValueError here
            # instead would let the most important rejection in the run escape as
            # an unhandled exception and take the whole analysis with it.
            return f"{text!r}: {exc}"


def _claim_type(value: Any) -> ClaimType:
    """A claim type, defaulting to ``reported`` rather than ``observed``.

    ``reported`` and not ``observed``: a claim whose type was not stated has been
    reported by something, and marking it ``observed`` would let it license a
    statistic later — which is exactly the inference the quality check refuses.
    """
    if isinstance(value, ClaimType):
        return value
    try:
        return ClaimType(str(value))
    except ValueError:
        return ClaimType.REPORTED


def _confidence(value: Any) -> Confidence:
    """A confidence, defaulting to low. Silence is not confidence."""
    if isinstance(value, Confidence):
        return value
    try:
        return Confidence(str(value))
    except ValueError:
        return Confidence.LOW
