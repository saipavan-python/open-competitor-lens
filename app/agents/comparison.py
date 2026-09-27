"""Turning the nine analysts' payloads into one comparison table.

The agent is thin because the judgement it must not get wrong is not the model's.
Which fields are comparable, what an absent value reads as, and whether a
dimension may be ranked at all are all decided in
:class:`~app.services.comparison_service.ComparisonService`, where the whole table
is visible. What is left here is turning nine differently-shaped payloads into
that service's field mapping, and refusing the step outright when there is not
enough to compare.

There is no model call. A comparison of four companies' published pricing is
arithmetic over what was retrieved, and a model in that loop could only introduce
a number nobody retrieved — the one thing the rest of the system exists to
prevent. Where judgement is genuinely required (which fields matter for this
comparison, and what to call them) the caller states it as labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.agents.base import Agent, AgentContext, AgentResult, AgentStatus
from app.core.logging import get_logger
from app.schemas.comparison import ComparisonMatrix
from app.services.comparison_service import ComparisonService

log = get_logger(__name__)


class ComparisonAgent(Agent):
    """Builds the cross-company comparison from the analysts' payloads."""

    name = "comparison"
    goal = "Put every company's findings side by side, with the gaps left visible."

    def __init__(
        self,
        *,
        service: ComparisonService | None = None,
        fields: Sequence[str] | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        self._service = service or ComparisonService()
        self._fields = list(fields) if fields is not None else None
        self._labels = dict(labels or {})

    async def run(self, ctx: AgentContext) -> AgentResult:
        """The comparison matrix, or a stated refusal to build one.

        Fewer than two companies is not a comparison, and building a table with
        one column would produce a document that looks like the deliverable and
        says nothing.
        """
        companies = self._companies(ctx)
        if len(companies) < 2:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.FAILED,
                errors=[
                    f"a comparison needs at least two companies; got {len(companies)}"
                ],
            )

        analyses = self._fields_by_company(ctx)
        try:
            matrix = self._service.build(
                companies, analyses, fields=self._fields, labels=self._labels
            )
        except TypeError as exc:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.FAILED,
                errors=[f"could not reduce the analyses: {exc}"],
            )

        matrix.notes.append(
            f"{len(matrix.companies)} companies compared across "
            f"{len(matrix.dimensions)} dimensions"
        )
        return AgentResult(
            agent=self.name,
            status=self._status(matrix),
            payload=matrix,
            warnings=self._warnings(matrix),
        )

    # -- inputs -----------------------------------------------------------

    def _companies(self, ctx: AgentContext) -> list[str]:
        """The companies in the comparison, target first and never duplicated."""
        listed = [str(name) for name in ctx.extras.get("companies", []) if str(name).strip()]
        target = ctx.company.strip()
        ordered: list[str] = []
        for name in ([target] if target else []) + listed:
            if name and name not in ordered:
                ordered.append(name)
        return ordered

    def _fields_by_company(self, ctx: AgentContext) -> dict[str, dict[str, Any]]:
        """Each company's comparable fields, reduced from its analysis payload.

        A payload the service cannot reduce is dropped to an empty mapping, which
        renders as a full row of sentinels — a visible gap. Passing it through
        would be worse only if a future reducer guessed, which is why
        ``fields_from`` raises on an unknown type instead.
        """
        raw = ctx.extras.get("analyses", {}) or {}
        fields: dict[str, dict[str, Any]] = {}
        for company, payload in raw.items():
            name = str(company)
            if not isinstance(payload, Mapping):
                try:
                    fields[name] = self._service.fields_from(payload)
                except TypeError:
                    log.warning(
                        "analysis payload could not be reduced",
                        extra={"company": name, "type": type(payload).__name__},
                    )
                    fields[name] = {}
                continue
            fields[name] = {
                key: value for key, value in payload.items() if key != "company"
            }
        return fields

    # -- reporting --------------------------------------------------------

    @staticmethod
    def _status(matrix: ComparisonMatrix) -> AgentStatus:
        """Partial when any company contributed nothing citable.

        Not failed: a table with gaps is still a table, and the gaps are the
        finding. Failing here would throw away a comparison that is correct about
        what it does know.
        """
        empty = [name for name, score in matrix.evidence_quality.items() if score == 0.0]
        return AgentStatus.PARTIAL if empty else AgentStatus.OK

    @staticmethod
    def _warnings(matrix: ComparisonMatrix) -> list[str]:
        """State which companies the table says nothing about."""
        empty = sorted(name for name, score in matrix.evidence_quality.items() if score == 0.0)
        if not empty:
            return []
        return [
            f"no cited findings for {', '.join(empty)}; "
            f"their cells read as not publicly available"
        ]
