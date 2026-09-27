"""The run's central invariant: if a claim carries a source id, we fetched it.

`ResearchService` is the only object in the system that knows what a run has
retrieved. Everything downstream — competitor validation, the analyst prompts,
the citation index, the final report — reads its state from here, so these tests
are about the ways state can go wrong quietly: a source that was never fetched
becoming citable, a failed retrieval disappearing, a claim bypassing the gate.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.constants import PageType, RetrievalStatus, SourceType
from app.core.exceptions import EvidenceNotSupportedError
from app.providers.base import SearchProvider
from app.schemas.evidence import ClaimType, Confidence
from app.schemas.search import SearchHit
from app.services.research_service import ResearchService
from app.services.search_service import SearchService


def _page(
    source_id: str = "src-1",
    *,
    url: str = "https://acme.example/pricing",
    text: str = "Starter plans start at $10 per month for one seat.",
    source_type: SourceType = SourceType.OFFICIAL_SITE,
    status: RetrievalStatus = RetrievalStatus.OK,
) -> object:
    from app.schemas.evidence import Source

    return Source(
        id=source_id,
        url=url,
        canonical_url=url,
        title="Pricing",
        text=text,
        domain="acme.example",
        source_type=source_type,
        page_type=PageType.PRICING,
        retrieval_status=status,
        retrieved_at=datetime.now(UTC),
    )


def _research() -> ResearchService:
    return ResearchService(company="Acme", search=SearchService(StubProvider()))


class StubProvider(SearchProvider):
    async def search(self, query: str, *, limit: int = 10, **kwargs) -> list[SearchHit]:
        return []


def _evidence_for_a_source_that_was_never_fetched() -> object:
    """Evidence pointing at a source id the run does not have.

    Built directly rather than through the gate, because the gate is exactly
    what prevents this state — and the index's own defence against it still has
    to be tested on its own.
    """
    from app.schemas.evidence import Evidence

    return Evidence(
        id="ev-orphan",
        claim="Acme was founded in 1999.",
        source_id="src-never-fetched",
        source_url="https://elsewhere.example/about",
        evidence_text="founded in 1999",
        claim_type=ClaimType.OBSERVED,
        confidence=Confidence.HIGH,
    )


class TestSourceRecording:
    def test_a_fetched_page_becomes_citable(self) -> None:
        research = _research()
        research.record_fetched(_page())
        assert [source.id for source in research.citable_sources()] == ["src-1"]

    def test_a_snippet_is_never_citable(self) -> None:
        """The whole point: search results can be found but not quoted."""
        research = _research()
        research.record_snippets([_page("snippet-1", text=None,
                                        source_type=SourceType.SEARCH_SNIPPET)])
        assert research.citable_sources() == []

    def test_a_failed_retrieval_is_kept_but_not_citable(self) -> None:
        """Kept for the audit trail; a page that did not load cannot back a claim."""
        research = _research()
        research.record_failure(
            _page("src-2", text=None, status=RetrievalStatus.FAILED),
            reason="connection reset",
        )
        assert research.source("src-2") is not None
        assert research.citable_sources() == []

    def test_a_failure_keeps_its_reason(self) -> None:
        research = _research()
        research.record_failure(
            _page("src-2", text=None, status=RetrievalStatus.FAILED),
            reason="connection reset",
        )
        assert "connection reset" in (research.source("src-2").notes or "")

    def test_a_page_with_no_text_is_not_citable(self) -> None:
        """Registered but empty: it has no quote to offer, so it cannot back one."""
        research = _research()
        research.record_fetched(_page("src-3", text=None))
        assert research.citable_sources() == []

    def test_recording_a_real_page_as_a_snippet_is_rejected(self) -> None:
        """Mislabeling a fetched page would smuggle it past the snippet rules."""
        research = _research()
        with pytest.raises(ValueError, match="must be recorded"):
            research.record_snippets([_page("src-1")])

    def test_sources_includes_both_kinds(self) -> None:
        research = _research()
        research.record_fetched(_page("src-1"))
        research.record_snippets(
            [_page("snippet-1", text=None, source_type=SourceType.SEARCH_SNIPPET)]
        )
        assert set(research.sources) == {"src-1", "snippet-1"}


class TestClaimGate:
    def test_a_supported_claim_is_attached(self) -> None:
        research = _research()
        page = research.record_fetched(_page())
        evidence = research.record_claim(
            "Starter plans start at $10 per month.",
            source=page,
            quote="Starter plans start at $10 per month",
            claim_type=ClaimType.OBSERVED,
            confidence=Confidence.HIGH,
        )
        assert evidence.claim

    def test_a_claim_cannot_cite_a_source_the_run_never_fetched(self) -> None:
        """The invariant, stated as a test because it is the system's spine."""
        research = _research()
        with pytest.raises(ValueError, match="not retrieved by this run"):
            research.record_claim(
                "Anything at all.",
                source=_page("never-fetched"),
                quote="anything",
                claim_type=ClaimType.OBSERVED,
                confidence=Confidence.HIGH,
            )

    def test_a_claim_cannot_cite_a_snippet(self) -> None:
        """Even a snippet this run genuinely found: a search result is not a page.

        The error names the structural reason, not a missing quote, so a caller
        learns the page was never retrieved rather than guessing that it fetched
        one and misread it.
        """
        research = _research()
        snippet = _page("snippet-1", text=None, source_type=SourceType.SEARCH_SNIPPET)
        research.record_snippets([snippet])
        with pytest.raises(ValueError, match="search snippet"):
            research.record_claim(
                "Acme is a hosting company.",
                source=snippet,
                quote="hosting",
                claim_type=ClaimType.REPORTED,
                confidence=Confidence.LOW,
            )

    def test_a_quote_that_is_not_in_the_page_is_rejected(self) -> None:
        """Verification is the gate's job and is not optional."""
        research = _research()
        page = research.record_fetched(_page())
        with pytest.raises(EvidenceNotSupportedError):
            research.record_claim(
                "Starter plans are free forever.",
                source=page,
                quote="free forever",
                claim_type=ClaimType.OBSERVED,
                confidence=Confidence.HIGH,
            )

    def test_an_unsupported_claim_is_recorded_not_dropped(self) -> None:
        """A refusal has to be visible, or the report silently omits a finding."""
        research = _research()
        research.record_unsupported("Acme raised a Series B.", "no source found")
        assert research.unsupported_claims()
        assert research.summary()["rejected_claims"] == 1

    def test_evidence_is_retrievable_by_claim_key(self) -> None:
        research = _research()
        page = research.record_fetched(_page())
        research.record_claim(
            "Starter plans start at $10 per month.",
            source=page,
            quote="Starter plans start at $10",
            claim_type=ClaimType.OBSERVED,
            confidence=Confidence.HIGH,
            claim_key="pricing.starter",
        )
        assert research.evidence_for("pricing.starter")


class TestCitations:
    def test_the_citation_index_is_keyed_by_the_backing_source(self) -> None:
        research = _research()
        page = research.record_fetched(_page())
        evidence = research.record_claim(
            "Starter plans start at $10 per month.",
            source=page,
            quote="Starter plans start at $10",
            claim_type=ClaimType.OBSERVED,
            confidence=Confidence.HIGH,
        )
        index = research.citation_index()
        assert index["src-1"] == [evidence.id]

    def test_a_snippet_backed_claim_leaves_no_citation(self) -> None:
        """An orphan is reported rather than published as a citation."""
        research = ResearchService(company="Acme", search=SearchService(StubProvider()))
        research._snippets["snippet-1"] = _page(
            "snippet-1", text=None, source_type=SourceType.SEARCH_SNIPPET
        )
        research._sources["src-1"] = _page("src-1")
        # Evidence naming a source the run never fetched must not be indexed.
        research.evidence._evidence.append(
            _evidence_for_a_source_that_was_never_fetched()
        )
        assert research.citation_index() == {}
        assert research.resolve_citations().orphans

    def test_a_citation_is_available_by_evidence_id(self) -> None:
        research = _research()
        page = research.record_fetched(_page())
        evidence = research.record_claim(
            "Starter plans start at $10 per month.",
            source=page,
            quote="Starter plans start at $10",
            claim_type=ClaimType.OBSERVED,
            confidence=Confidence.HIGH,
        )
        citation = research.citation_for(evidence.id)
        assert citation is not None
        assert citation.source_id == "src-1"

    def test_an_unknown_evidence_id_yields_no_citation(self) -> None:
        research = _research()
        research.record_fetched(_page())
        assert research.citation_for("no-such-evidence") is None


class TestSummary:
    def test_the_summary_counts_what_the_run_knows(self) -> None:
        research = _research()
        page = research.record_fetched(_page("src-1"))
        research.record_snippets(
            [_page("snippet-1", text=None, source_type=SourceType.SEARCH_SNIPPET)]
        )
        research.record_claim(
            "Starter plans start at $10 per month.",
            source=page,
            quote="Starter plans start at $10",
            claim_type=ClaimType.OBSERVED,
            confidence=Confidence.HIGH,
        )
        summary = research.summary()
        assert summary["sources"] == 1
        assert summary["citable"] == 1
        assert summary["snippets"] == 1
        assert summary["evidence"] == 1
