"""Turning an observed gap into an opportunity, without inventing one.

An opportunity is the only thing in a competitor report that tells a reader what
to *do*, so it is the easiest thing to fabricate and the most expensive when
fabricated. The usual failure is a sentence like "Acme should offer a free tier"
generated because free tiers are where the market is heading — plausible,
actionable, and backed by nothing.

So nothing here is generated. Every opportunity this agent emits is a gap the
comparison actually observed, carrying the source ids of the evidence that
revealed it:

* a competitor publishes something the target does not, and the evidence for it
  exists;
* the target is silent on a field where both competitors had pages to publish on;
* a price moved in a direction the run can evidence.

If the evidence is not there, there is no opportunity, and the absence is
reported rather than filled. :class:`~app.schemas.market.Opportunity` enforces the
same rule at construction, so a future caller cannot route around this agent by
building one directly — the refusal is a property of the data, not of this code.

No model is involved. The interesting judgement here is which gaps are worth
naming, and inventing that judgement is precisely the risk.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.agents.base import Agent, AgentContext, AgentResult, AgentStatus
from app.core.logging import get_logger
from app.schemas.comparison import ComparisonMatrix
from app.schemas.market import Opportunity
from app.schemas.pricing import PricingAnalysis

log = get_logger(__name__)

#: Fields whose absence in the target, beside a competitor that has them, is a
#: gap worth naming. Deliberately a short list of commercial fields: a missing
#: ``meta_description`` is an SEO task, not a competitive opportunity, and
#: listing every comparable field would bury the three that matter.
OPPORTUNITY_FIELDS: tuple[str, ...] = (
    "has_free_tier",
    "has_trial",
    "has_enterprise_tier",
    "billing_periods",
    "differentiators",
    "target_segments",
)

#: The field each gap is reported under, so several gaps in one category read as
#: one opportunity rather than several.
CATEGORY_OF: dict[str, str] = {
    "has_free_tier": "pricing",
    "has_trial": "pricing",
    "has_enterprise_tier": "pricing",
    "billing_periods": "pricing",
    "differentiators": "product",
    "target_segments": "website",
}


class OpportunityAgent(Agent):
    """Derives opportunities from observed gaps, each carrying its evidence."""

    name = "opportunity"
    goal = "Name what the target could do differently, with the evidence that shows the gap."

    def __init__(self, *, fields: Sequence[str] = OPPORTUNITY_FIELDS) -> None:
        self._fields = tuple(fields)

    async def run(self, ctx: AgentContext) -> AgentResult:
        """The opportunities the evidence supports, and a count of those it did not.

        Nothing retrieved means no opportunities and a partial result — an
        unbroken run that happens to have found nothing to say is not a finding
        about the market.
        """
        matrix = ctx.extras.get("comparison")
        evidence_index = ctx.extras.get("evidence_by_company", {}) or {}
        if not isinstance(matrix, ComparisonMatrix):
            return AgentResult(
                agent=self.name,
                status=AgentStatus.FAILED,
                errors=["no comparison matrix was supplied; opportunities are read from it"],
            )

        opportunities, skipped = self._from_gaps(matrix, evidence_index, ctx.company)
        if not opportunities:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.PARTIAL,
                warnings=["no observed gap was backed by evidence, so no opportunity is offered"],
            )

        for opportunity in opportunities:
            log.info(
                "opportunity derived from an observed gap",
                extra={"category": opportunity.category, "evidence": len(opportunity.evidence)},
            )
        return AgentResult(
            agent=self.name,
            status=AgentStatus.OK,
            payload=opportunities,
            evidence_ids=sorted({ref for item in opportunities for ref in item.evidence}),
            warnings=(
                [
                    f"{skipped} observed gap(s) had no evidence and were not "
                    f"turned into opportunities"
                ]
                if skipped
                else []
            ),
        )

    # -- the derivation ---------------------------------------------------

    def _from_gaps(
        self,
        matrix: ComparisonMatrix,
        evidence_index: Mapping[str, Any],
        target: str,
    ) -> tuple[list[Opportunity], int]:
        """Every evidenced gap, and how many gaps were dropped for lack of evidence."""
        opportunities: list[Opportunity] = []
        skipped = 0
        competitors = [name for name in matrix.companies if name != target]

        for field in self._fields:
            row = matrix.row(field)
            if row is None:
                continue
            target_cell = row.cells.get(target)
            if target_cell is None or target_cell.is_known:
                continue
            for competitor in competitors:
                cell = row.cells.get(competitor)
                if cell is None or not cell.is_known:
                    continue
                evidence = self._evidence_for(evidence_index, competitor, field)
                if not evidence:
                    skipped += 1
                    continue
                opportunities.append(
                    self._opportunity(field, competitor, cell.display, evidence)
                )
        return opportunities, skipped

    @staticmethod
    def _opportunity(
        field: str, competitor: str, value: str, evidence: list[str]
    ) -> Opportunity:
        """One opportunity, phrased as the observation that produced it.

        Stating the gap rather than the recommendation is deliberate: the reader
        can see what was observed and disagree with the inference, and the schema
        then refuses to let the object exist at all without the evidence behind
        that observation.
        """
        readable = field.replace("_", " ")
        return Opportunity(
            opportunity=f"{competitor} publishes {readable} ({value}) where the target does not",
            reasoning=(
                f"the comparison shows {competitor} with a value for {readable} and no "
                f"publicly available value for the target; the evidence below is what "
                f"the run retrieved"
            ),
            evidence=evidence,
            affected_competitors=[competitor],
            category=CATEGORY_OF.get(field),
        )

    @staticmethod
    def _evidence_for(
        evidence_index: Mapping[str, Any], competitor: str, field: str
    ) -> list[str]:
        """The evidence ids backing one competitor's value for one field.

        Looked up per competitor and per field, so an opportunity is never backed
        by evidence about a different company or a different subject. Returns an
        empty list rather than a best guess, and the caller skips the opportunity.
        """
        per_company = evidence_index.get(competitor)
        if not isinstance(per_company, Mapping):
            return []
        by_field = per_company.get(field)
        if isinstance(by_field, str):
            return [by_field] if by_field.strip() else []
        if isinstance(by_field, Iterable):
            return [str(item) for item in by_field if str(item).strip()]
        return []


def pricing_evidence(analysis: PricingAnalysis) -> list[str]:
    """The evidence ids a pricing analysis cited, for indexing by field."""
    return [item for item in analysis.evidence if item]
