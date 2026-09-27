"""The per-run object everything else hangs off.

A run accumulates sources, claims, and competitors over a long, branching,
resumable execution. The temptation is for each component to keep its own copy
and reconcile later. That reconciliation is where claims end up citing a source
the run never fetched, because two components disagreed about what "known" meant
and one of them was wrong in a way nobody notices until publication.

So this service is the single owner of a run's research state, and it is
deliberately thin. It does not fetch, does not judge, and does not decide what is
true. It holds the sources this run actually retrieved, and it makes every claim
pass through :class:`~app.services.evidence_service.EvidenceService` on the way
in.

**The only way to add a source is to record that it was fetched.** Search hits
are registered separately, and as :attr:`SourceType.SEARCH_SNIPPET`, precisely
so they sit outside the citable set. That is the single invariant the rest of the
system relies on: if a claim carries a source id, this run retrieved that page.

A failed retrieval is recorded as a failure, with its status, rather than
omitted. Dropping it would make "we looked and found nothing" indistinguishable
from "we never looked", which is the difference between a finding and a gap.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from app.core.constants import SourceType
from app.core.logging import get_logger
from app.schemas.competitor import ValidatedCompetitor
from app.schemas.evidence import Citation, ClaimType, Confidence, Evidence, Source
from app.services.citation_service import CitationResolution, CitationService
from app.services.evidence_service import EvidenceService
from app.services.search_service import SearchService

log = get_logger(__name__)


@dataclass
class ResearchService:
    """Everything one run knows, and the only door it knows it through."""

    company: str
    search: SearchService
    evidence: EvidenceService = field(default_factory=EvidenceService)
    citations: CitationService = field(default_factory=CitationService)
    competitors: list[ValidatedCompetitor] = field(default_factory=list)

    _sources: dict[str, Source] = field(default_factory=dict, repr=False)
    _snippets: dict[str, Source] = field(default_factory=dict, repr=False)

    # -- sources ----------------------------------------------------------

    def record_fetched(self, source: Source) -> Source:
        """Record a page this run retrieved, making it citable."""
        stored = self.evidence.register_fetched(source)
        self._sources[stored.id] = stored
        return stored

    def record_fetched_many(self, sources: Iterable[Source]) -> list[Source]:
        return [self.record_fetched(source) for source in sources]

    def record_failure(
        self, source: Source, *, reason: str | None = None
    ) -> Source:
        """Record a retrieval that did not succeed.

        Kept, with its status, so the run can report the difference between a
        page that was absent and a page that was never requested. Neither is
        citable, which is the point — the entry exists for the audit trail, not
        to be cited.
        """
        if reason:
            source.notes = "; ".join(filter(None, [source.notes, reason]))
        self._sources.setdefault(source.id, source)
        return source

    def record_snippets(self, sources: Iterable[Source]) -> list[Source]:
        """Record search hits as discovery artefacts, outside the citable set."""
        stored: list[Source] = []
        for source in sources:
            if source.source_type is not SourceType.SEARCH_SNIPPET:
                raise ValueError(
                    f"{source.id!r} is a {source.source_type} and must be recorded "
                    f"as fetched or as a failure, not as a snippet"
                )
            self._snippets[source.id] = source
            stored.append(source)
        return stored

    @property
    def sources(self) -> Mapping[str, Source]:
        """Every source this run knows, citable or not."""
        return {**self._snippets, **self._sources}

    def citable_sources(self) -> list[Source]:
        """Sources eligible to back a claim: fetched, and not a snippet.

        A page that failed is excluded by status, and a snippet is excluded by
        type, so neither can reach a claim even if something upstream hands one
        over by mistake.
        """
        return [
            source
            for source in self._sources.values()
            if source.source_type is not SourceType.SEARCH_SNIPPET and source.text
        ]

    def source(self, source_id: str) -> Source | None:
        return self.sources.get(source_id)

    # -- claims -----------------------------------------------------------

    def record_claim(
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
        """Attach a claim, or raise.

        There is no ``try`` variant and no way to bypass the gate, by design.
        A caller that cannot support a claim calls :meth:`record_unsupported`,
        which records the refusal as a visible outcome. Silently dropping an
        unverifiable claim is the failure mode this system exists to prevent, so
        it has no code path.

        Snippets are rejected here rather than left to the evidence gate. A
        snippet has no page text, so the gate would reject the quote anyway, but
        as a *content* failure — which reads as "that quote was not on the page"
        rather than the structural truth, "that page was never retrieved".
        """
        if source.source_type is SourceType.SEARCH_SNIPPET or source.id in self._snippets:
            raise ValueError(
                f"source {source.id!r} is a search snippet, not a retrieved page; "
                f"a claim cannot cite it"
            )
        resolved = self._sources.get(source.id)
        if resolved is None:
            raise ValueError(
                f"source {source.id!r} was not retrieved by this run; a claim "
                f"cannot cite it"
            )
        return self.evidence.attach(
            claim,
            source=resolved,
            quote=quote,
            claim_type=claim_type,
            confidence=confidence,
            claim_key=claim_key,
            competitor=competitor,
            category=category,
        )

    def record_unsupported(self, claim: str, reason: str) -> str:
        """Record a claim that could not be supported, and why."""
        return self.evidence.reject(claim, reason)

    def unsupported_claims(self) -> list[str]:
        """Claims the run wanted to make and could not.

        Delegated to the evidence service rather than tracked here, so the two
        cannot disagree about what was rejected.
        """
        return self.evidence.unsupported_claims()

    def evidence_for(self, claim_key: str) -> list[Evidence]:
        return [
            item
            for item in self.evidence.all_evidence()
            if item.claim_key == claim_key
        ]

    # -- publication ------------------------------------------------------

    def citation_index(self) -> dict[str, list[str]]:
        """Evidence ids keyed by the source that backs them.

        Built from :meth:`citable_sources` only, so anything the run could not
        legitimately cite is already excluded before the index is built.
        """
        return self.citations.build_index(
            self.citable_sources(), self.evidence.all_evidence()
        )

    def resolve_citations(self) -> CitationResolution:
        """Resolve every claim's evidence to citations, flagging anything orphaned."""
        return self.citations.resolve(self.citation_index(), list(self.sources.values()))

    def citation_for(self, evidence_id: str) -> Citation | None:
        """One citation by evidence id, or ``None`` if it cannot be made."""
        for item in self.evidence.all_evidence():
            if item.id != evidence_id:
                continue
            source = self.source(item.source_id)
            if source is None:
                return None
            return self.citations.to_citation(item, source)
        return None

    # -- competitors ------------------------------------------------------

    def set_competitors(self, competitors: Iterable[ValidatedCompetitor]) -> None:
        self.competitors = list(competitors)

    def add_competitor(self, competitor: ValidatedCompetitor) -> None:
        self.competitors.append(competitor)

    def summary(self) -> dict[str, int]:
        """Counts a run can report about itself."""
        return {
            "sources": len(self._sources),
            "citable": len(self.citable_sources()),
            "snippets": len(self._snippets),
            "evidence": len(self.evidence.all_evidence()),
            "rejected_claims": len(self.unsupported_claims()),
            "competitors": len(self.competitors),
        }
