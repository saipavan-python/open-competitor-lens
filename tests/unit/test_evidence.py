"""The evidence contract: nothing is asserted without a retrieved source.

This is the part of the system that decides whether the report is worth
believing, so the tests here are adversarial rather than happy-path. Each one
describes a way a language model might produce something plausible and
unsupported, and asserts that the evidence layer refuses it.

The three failures being guarded against, in order of how badly they would
corrupt a report:

1. A citation to a source the run never fetched (a hallucinated URL).
2. A quote that does not appear in the retrieved page (a fabricated quote).
3. Confidence that outranks the claim type, i.e. asserting as fact something the
   source never actually said.
"""

from __future__ import annotations

import pytest

from app.core.constants import ClaimType, Confidence, RetrievalStatus, SourceType
from app.core.exceptions import EvidenceNotSupportedError
from app.schemas.evidence import Citation, Evidence, Source
from app.services.citation_service import CitationService
from app.services.evidence_service import EvidenceService

PAGE_TEXT = (
    "Acme Cloud plans start at $10 per seat per month. The Enterprise plan "
    "includes SSO, audit logs and a 99.9% uptime SLA. Free tier is limited "
    "to 100 events per month."
)


def _source(
    source_id: str = "src-1",
    *,
    url: str = "https://acme.com/pricing",
    text: str | None = PAGE_TEXT,
    status: RetrievalStatus = RetrievalStatus.OK,
    source_type: SourceType = SourceType.OFFICIAL_SITE,
    company: str = "Acme",
) -> Source:
    return Source(
        id=source_id, url=url, text=text, retrieval_status=status,
        source_type=source_type, company=company, title="Pricing",
    )


@pytest.fixture()
def service() -> EvidenceService:
    return EvidenceService()


@pytest.fixture()
def run(service: EvidenceService) -> EvidenceService:
    """A service that has 'fetched' one known source."""
    service.register_fetched(_source())
    return service


class TestTraceability:
    def test_a_claim_citing_an_unfetched_source_is_refused(
        self, service: EvidenceService
    ) -> None:
        with pytest.raises(EvidenceNotSupportedError) as excinfo:
            service.attach(
                "Acme charges $10 per seat", source=_source("src-never-fetched"),
                quote="start at $10 per seat", claim_type=ClaimType.OBSERVED,
                confidence=Confidence.HIGH,
            )
        assert "src-never-fetched" in str(excinfo.value)

    def test_a_claim_citing_a_known_source_is_accepted(
        self, run: EvidenceService
    ) -> None:
        evidence = run.attach(
            "Acme charges $10 per seat", source=_source(),
            quote="start at $10 per seat", claim_type=ClaimType.OBSERVED,
            confidence=Confidence.HIGH,
        )
        assert evidence.source_id == "src-1"
        assert evidence.source_url == "https://acme.com/pricing"
        assert evidence.claim_type is ClaimType.OBSERVED

    def test_the_stored_text_is_authoritative_not_the_caller_s_copy(
        self, run: EvidenceService
    ) -> None:
        """The caller cannot smuggle in text the run never retrieved."""
        forged = _source(text="Totally different content")
        with pytest.raises(EvidenceNotSupportedError):
            run.attach(
                "Anything", source=forged, quote="start at $10 per seat",
                claim_type=ClaimType.OBSERVED, confidence=Confidence.MEDIUM,
            )

    def test_a_source_that_failed_retrieval_cannot_support_anything(
        self, service: EvidenceService
    ) -> None:
        failed = _source("src-2", status=RetrievalStatus.FAILED, text=None)
        service.register_fetched(failed)
        with pytest.raises(EvidenceNotSupportedError) as excinfo:
            service.attach(
                "Acme charges $10", source=failed, quote="start at $10",
                claim_type=ClaimType.OBSERVED, confidence=Confidence.LOW,
            )
        assert "failed" in str(excinfo.value).lower()

    def test_a_blocked_source_cannot_support_anything(self, service: EvidenceService
                                                      ) -> None:
        blocked = _source("src-3", status=RetrievalStatus.BLOCKED, text=None)
        service.register_fetched(blocked)
        with pytest.raises(EvidenceNotSupportedError):
            service.attach(
                "Acme charges $10", source=blocked, quote="start at $10",
                claim_type=ClaimType.OBSERVED, confidence=Confidence.LOW,
            )

    def test_a_source_with_no_text_cannot_be_quoted(self, run: EvidenceService) -> None:
        empty = _source("src-4", text=None)
        run.register_fetched(empty)
        with pytest.raises(EvidenceNotSupportedError):
            run.attach(
                "Acme charges $10", source=empty, quote="start at $10",
                claim_type=ClaimType.OBSERVED, confidence=Confidence.LOW,
            )


class TestQuoteVerification:
    def test_a_quote_absent_from_the_page_is_refused(self, run: EvidenceService) -> None:
        with pytest.raises(EvidenceNotSupportedError) as excinfo:
            run.attach(
                "Acme offers 99.99% uptime", source=_source(),
                quote="99.99% uptime guarantee", claim_type=ClaimType.OBSERVED,
                confidence=Confidence.HIGH,
            )
        assert "quote" in str(excinfo.value).lower()

    def test_a_quote_with_different_whitespace_is_accepted(
        self, run: EvidenceService
    ) -> None:
        """HTML extraction reflows spacing; that is not a fabricated quote."""
        evidence = run.attach(
            "Acme has an SLA", source=_source(),
            quote="includes   SSO,\naudit logs", claim_type=ClaimType.OBSERVED,
            confidence=Confidence.MEDIUM,
        )
        assert evidence.evidence_text

    def test_a_quote_differing_only_in_case_is_accepted(self, run: EvidenceService
                                                        ) -> None:
        evidence = run.attach(
            "Acme has SSO", source=_source(), quote="INCLUDES SSO",
            claim_type=ClaimType.OBSERVED, confidence=Confidence.MEDIUM,
        )
        assert evidence.evidence_text

    def test_a_source_type_cannot_be_claimed_by_the_caller(self, run: EvidenceService
                                                          ) -> None:
        """Authority is derived from what was fetched, not from the claim."""
        secondary = _source("src-5", url="https://blog.example.com/x",
                            source_type=SourceType.INDEPENDENT)
        run.register_fetched(secondary)
        evidence = run.attach(
            "Something", source=secondary, quote="Acme Cloud plans",
            claim_type=ClaimType.OBSERVED, confidence=Confidence.LOW,
        )
        assert evidence.source_type is SourceType.INDEPENDENT


class TestConfidenceRules:
    def test_an_inference_may_not_carry_high_confidence(
        self, run: EvidenceService
    ) -> None:
        with pytest.raises(EvidenceNotSupportedError) as excinfo:
            run.attach(
                "Acme is enterprise-grade", source=_source(),
                quote="includes SSO, audit logs", claim_type=ClaimType.INFERRED,
                confidence=Confidence.HIGH,
            )
        assert "inference" in str(excinfo.value).lower()

    def test_an_inference_with_medium_confidence_is_accepted(
        self, run: EvidenceService
    ) -> None:
        evidence = run.attach(
            "Acme is enterprise-grade", source=_source(),
            quote="includes SSO, audit logs", claim_type=ClaimType.INFERRED,
            confidence=Confidence.MEDIUM,
        )
        assert evidence.claim_type is ClaimType.INFERRED

    def test_high_confidence_requires_a_primary_source(self, service: EvidenceService
                                                       ) -> None:
        snippet = _source("src-6", url="https://search.example.com/?q=acme",
                          source_type=SourceType.SEARCH_SNIPPET)
        service.register_fetched(snippet)
        with pytest.raises(EvidenceNotSupportedError) as excinfo:
            service.attach(
                "Acme charges $10", source=snippet, quote="Acme Cloud plans start",
                claim_type=ClaimType.OBSERVED, confidence=Confidence.HIGH,
            )
        assert "primary" in str(excinfo.value).lower()

    def test_an_unknown_claim_type_needs_an_explicit_unknown(self
                                                             , run: EvidenceService
                                                             ) -> None:
        with pytest.raises(EvidenceNotSupportedError):
            run.attach(
                "Acme's market share", source=_source(), quote="start at $10",
                claim_type=ClaimType.UNKNOWN, confidence=Confidence.HIGH,
            )


class TestConflicts:
    def test_disagreement_is_preserved_as_two_evidences_and_flagged(
        self, run: EvidenceService
    ) -> None:
        """A conflict is information. Silently picking a side is a lie."""
        official = _source("src-a")
        run.register_fetched(official)
        other = _source("src-b", url="https://news.example.com/item",
                        text="Acme Cloud plans start at $20 per seat per month.",
                        source_type=SourceType.NEWS, company="Acme")
        run.register_fetched(other)
        first = run.attach(
            "Acme starts at $10 per seat", source=official,
            quote="start at $10 per seat", claim_type=ClaimType.OBSERVED,
            confidence=Confidence.HIGH, claim_key="acme.base_price",
        )
        second = run.attach(
            "Acme starts at $20 per seat", source=other,
            quote="start at $20 per seat", claim_type=ClaimType.OBSERVED,
            confidence=Confidence.MEDIUM, claim_key="acme.base_price",
        )
        assert first.id != second.id
        assert first.conflicts_with == [second.id]
        assert second.conflicts_with == [first.id]

    def test_agreement_raises_confidence_instead_of_conflicting(
        self, run: EvidenceService
    ) -> None:
        second_source = _source("src-b", url="https://acme.com/pricing-detail",
                                text=PAGE_TEXT, company="Acme")
        run.register_fetched(second_source)
        first = run.attach(
            "Acme starts at $10 per seat", source=_source(),
            quote="start at $10 per seat", claim_type=ClaimType.OBSERVED,
            confidence=Confidence.MEDIUM, claim_key="acme.base_price",
        )
        second = run.attach(
            "Acme starts at $10 per seat", source=second_source,
            quote="start at $10 per seat", claim_type=ClaimType.OBSERVED,
            confidence=Confidence.HIGH, claim_key="acme.base_price",
        )
        assert not first.conflicts_with
        assert not second.conflicts_with

    def test_claims_with_different_keys_never_conflict(self, run: EvidenceService
                                                       ) -> None:
        run.attach("A", source=_source(), quote="start at $10",
                   claim_type=ClaimType.OBSERVED, confidence=Confidence.MEDIUM,
                   claim_key="acme.price")
        other = run.attach("B", source=_source(), quote="99.9% uptime",
                           claim_type=ClaimType.OBSERVED, confidence=Confidence.MEDIUM,
                           claim_key="acme.sla")
        assert not other.conflicts_with


class TestBundles:
    def test_a_bundle_summarises_its_evidence(self, run: EvidenceService) -> None:
        run.attach("Acme starts at $10", source=_source(), quote="start at $10",
                   claim_type=ClaimType.OBSERVED, confidence=Confidence.HIGH,
                   claim_key="acme.price")
        bundle = run.bundle_for("acme.price")
        assert bundle is not None
        assert bundle.best_confidence is Confidence.HIGH
        assert bundle.claims_supported

    def test_a_bundle_for_an_unknown_key_is_none(self, run: EvidenceService) -> None:
        assert run.bundle_for("never.claimed") is None


class TestCitationService:
    def test_the_index_maps_each_source_to_its_claims(self, run: EvidenceService
                                                       ) -> None:
        one = run.attach("Acme starts at $10", source=_source(), quote="start at $10",
                         claim_type=ClaimType.OBSERVED, confidence=Confidence.HIGH)
        run.attach("Acme has SSO", source=_source(), quote="includes SSO",
                   claim_type=ClaimType.OBSERVED, confidence=Confidence.MEDIUM)
        index = CitationService().build_index([_source()], [one, *run.all_evidence()])
        assert index["src-1"]
        assert one.id in index["src-1"]

    def test_an_evidence_citing_an_absent_source_is_flagged(self) -> None:
        evidence = Evidence(
            id="ev-1", claim="Something", source_id="src-missing",
            source_url="https://acme.com/x", claim_type=ClaimType.OBSERVED,
            confidence=Confidence.MEDIUM,
        )
        index = CitationService().build_index([_source()], [evidence])
        assert index == {}

    def test_resolution_returns_only_cited_sources(self) -> None:
        cited, uncited = _source("src-1"), _source("src-2",
                                                   url="https://acme.com/about")
        index = {"src-1": ["ev-1"]}
        resolved, orphans = CitationService().resolve(index, [cited, uncited])
        assert [item.id for item in resolved] == ["src-1"]
        assert [item.id for item in orphans] == ["src-2"]

    def test_a_citation_carries_the_quote_and_the_url(self, run: EvidenceService
                                                      ) -> None:
        evidence = run.attach("Acme starts at $10", source=_source(),
                              quote="start at $10 per seat",
                              claim_type=ClaimType.OBSERVED, confidence=Confidence.HIGH)
        citation = CitationService().to_citation(evidence, _source())
        assert isinstance(citation, Citation)
        assert citation.url == "https://acme.com/pricing"
        assert citation.quote == "start at $10 per seat"
        assert citation.source_type is SourceType.OFFICIAL_SITE

    def test_a_citation_for_an_unknown_evidence_is_refused(self) -> None:
        from app.core.exceptions import ValidationError

        with pytest.raises(ValidationError):
            CitationService().to_citation(None, _source())


class TestEvidenceServiceBookkeeping:
    def test_registering_a_source_twice_is_harmless(self, run: EvidenceService) -> None:
        run.register_fetched(_source())
        run.register_fetched(_source())
        assert len(run) == 1

    def test_reset_clears_the_run(self, run: EvidenceService) -> None:
        run.attach("Acme starts at $10", source=_source(), quote="start at $10",
                   claim_type=ClaimType.OBSERVED, confidence=Confidence.HIGH)
        run.reset()
        assert len(run) == 0
        assert run.all_evidence() == []

    def test_an_untested_claim_cannot_appear(self, run: EvidenceService) -> None:
        """Nothing enters the evidence list without passing attach()."""
        assert run.all_evidence() == []
