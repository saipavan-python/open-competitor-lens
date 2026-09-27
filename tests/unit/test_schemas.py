"""RED: domain schema contracts.

These tests pin the anti-hallucination invariants that the rest of the system
relies on: an unknown value is representable, a high-confidence claim must name
a source, and an inference can never be dressed up as an observation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.constants import (
    AnalysisCategory,
    ClaimType,
    Confidence,
    InsightType,
    PageType,
    RetrievalStatus,
    SourceType,
)
from app.schemas.common import UNKNOWN_NOT_AVAILABLE
from app.schemas.company import CompanyIdentity, CompanyProfile
from app.schemas.comparison import ComparisonCell, ComparisonMatrix
from app.schemas.competitor import CompetitorCandidate, ValidatedCompetitor
from app.schemas.evidence import Evidence, Source
from app.schemas.finding import Finding, Insight
from app.schemas.market import MarketInsight
from app.schemas.marketing import MarketingAnalysis
from app.schemas.pricing import PricingAnalysis, PricingPlan
from app.schemas.products import ProductAnalysis, ProductOffering
from app.schemas.report import ReportBundle, ReportMetadata
from app.schemas.research import (
    ReportConfig,
    ResearchPlan,
    ResearchQuestion,
    StartAnalysisRequest,
)
from app.schemas.reviews import ReviewAnalysis, ReviewTheme
from app.schemas.seo import SeoAnalysis
from app.schemas.social import SocialProfile
from app.schemas.technology import TechnologyAnalysis, TechnologySignal


def _source(**overrides: object) -> Source:
    payload: dict[str, object] = {
        "id": "src-1",
        "url": "https://acme.com/pricing",
        "canonical_url": "https://acme.com/pricing",
        "title": "Pricing - Acme",
        "source_type": SourceType.OFFICIAL_SITE,
        "domain": "acme.com",
        "company": "Acme",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "retrieval_status": RetrievalStatus.OK,
    }
    payload.update(overrides)
    return Source(**payload)  # type: ignore[arg-type]


class TestSource:
    def test_defaults_to_unknown_source_type(self) -> None:
        assert _source(source_type=None).source_type is SourceType.UNKNOWN

    def test_requires_a_url(self) -> None:
        with pytest.raises(ValidationError):
            _source(url=None)

    def test_derives_domain_when_absent(self) -> None:
        assert _source(domain=None).domain == "acme.com"

    def test_round_trips_through_json(self) -> None:
        restored = Source.model_validate_json(_source().model_dump_json())
        assert restored.id == "src-1"

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            _source(unexpected_field="x")


class TestEvidence:
    def test_rejects_high_confidence_without_a_source(self) -> None:
        with pytest.raises(ValidationError):
            Evidence(
                id="ev-1", claim="Acme charges $10", source_id="", source_url="",
                claim_type=ClaimType.OBSERVED, confidence=Confidence.HIGH,
                retrieved_at="2026-01-01T00:00:00Z",
            )

    def test_rejects_claim_type_guessed(self) -> None:
        with pytest.raises(ValidationError):
            Evidence(
                id="ev-1", claim="x", source_id="src-1", source_url="https://a.com",
                claim_type="guessed", confidence=Confidence.LOW,
                retrieved_at="2026-01-01T00:00:00Z",
            )

    def test_inference_cannot_be_high_confidence(self) -> None:
        with pytest.raises(ValidationError):
            Evidence(
                id="ev-1", claim="x", source_id="src-1", source_url="https://a.com",
                claim_type=ClaimType.INFERRED, confidence=Confidence.HIGH,
                retrieved_at="2026-01-01T00:00:00Z",
            )

    def test_unknown_claim_type_is_accepted(self) -> None:
        ev = Evidence(
            id="ev-1", claim="pricing not published", source_id="src-1",
            source_url="https://a.com", claim_type=ClaimType.UNKNOWN,
            confidence=Confidence.LOW, retrieved_at="2026-01-01T00:00:00Z",
        )
        assert ev.claim_type is ClaimType.UNKNOWN

    def test_empty_evidence_text_is_allowed_but_kept_empty(self) -> None:
        ev = Evidence(
            id="ev-1", claim="x", source_id="src-1", source_url="https://a.com",
            evidence_text=None, claim_type=ClaimType.OBSERVED,
            confidence=Confidence.MEDIUM, retrieved_at="2026-01-01T00:00:00Z",
        )
        assert ev.evidence_text is None

    def test_confidence_ordering_helper(self) -> None:
        assert Confidence.HIGH != Confidence.MEDIUM


class TestPricing:
    def test_hidden_pricing_is_preserved_not_invented(self) -> None:
        plan = PricingPlan(name="Enterprise", price=None, price_display="Contact sales")
        assert plan.price is None
        assert plan.price_display == "Contact sales"

    def test_public_price_keeps_original_display(self) -> None:
        plan = PricingPlan(name="Pro", price=29.0, currency="USD",
                           billing_period="monthly", price_display="$29/mo")
        assert plan.price == 29.0
        assert plan.price_display == "$29/mo"

    def test_analysis_defaults_unknown_fields_to_sentinel(self) -> None:
        analysis = PricingAnalysis(company="Acme", plans=[])
        assert analysis.pricing_model == UNKNOWN_NOT_AVAILABLE
        assert analysis.evidence == []

    def test_rejects_negative_price(self) -> None:
        with pytest.raises(ValidationError):
            PricingPlan(name="Bad", price=-5)


class TestProductsAndMarketing:
    def test_product_offering_keeps_unknown_differentiators(self) -> None:
        offering = ProductOffering(name="Widget")
        assert offering.features == []
        assert offering.differentiators == []

    def test_product_analysis_requires_company(self) -> None:
        with pytest.raises(ValidationError):
            ProductAnalysis(company="")

    def test_marketing_separates_observed_from_inferred(self) -> None:
        analysis = MarketingAnalysis(company="Acme", observations=["Hero CTA: Book a demo"])
        assert analysis.observations == ["Hero CTA: Book a demo"]
        assert analysis.inferences == []


class TestSocialReviewsTechnology:
    def test_social_profile_never_invents_metrics(self) -> None:
        profile = SocialProfile(platform="linkedin", profile_url="https://linkedin.com/company/acme")
        assert profile.public_metrics is None

    def test_rejects_invented_platform(self) -> None:
        with pytest.raises(ValidationError):
            SocialProfile(platform="myspace", profile_url="https://a.com")

    def test_review_theme_defaults_to_no_count(self) -> None:
        theme = ReviewTheme(theme="slow support", sentiment="negative")
        assert theme.mention_count == 0

    def test_rejects_invalid_sentiment(self) -> None:
        with pytest.raises(ValidationError):
            ReviewTheme(theme="x", sentiment="mixedish")

    def test_technology_signal_requires_a_known_category(self) -> None:
        with pytest.raises(ValidationError):
            TechnologySignal(technology="Rust", category="quantum", confidence="low")

    def test_technology_signal_accepts_known_category(self) -> None:
        signal = TechnologySignal(technology="Next.js", category="frontend",
                                  confidence=Confidence.MEDIUM, basis="public job posting")
        assert signal.category == "frontend"

    def test_analyses_accept_empty_category_envelope(self) -> None:
        assert SeoAnalysis(company="Acme").structured_data_found is False
        assert ReviewAnalysis(company="Acme").themes == []
        assert TechnologyAnalysis(company="Acme").signals == []


class TestCompanyAndCompetitor:
    def test_company_identity_normalises_aliases(self) -> None:
        identity = CompanyIdentity(canonical_name="Acme Ltd", aliases=["Acme", "acme.com"],
                                   domains=["acme.com"])
        assert identity.canonical_name == "Acme Ltd"

    def test_company_profile_defaults_confidence_low(self) -> None:
        profile = CompanyProfile(name="Acme")
        assert profile.confidence is Confidence.LOW
        assert profile.official_website is None

    def test_candidate_is_not_yet_validated(self) -> None:
        candidate = CompetitorCandidate(name="Beta", website="https://beta.com")
        assert candidate.validated is False

    def test_validated_competitor_demands_a_relevance_reason(self) -> None:
        with pytest.raises(ValidationError):
            ValidatedCompetitor(name="Beta", website="https://beta.com", validated=True,
                                relevance_reason="")

    def test_validated_competitor_carries_evidence(self) -> None:
        vc = ValidatedCompetitor(name="Beta", website="https://beta.com", validated=True,
                                 relevance_reason="listed as alternative on acme.com/compare",
                                 evidence=["ev-1"])
        assert vc.evidence == ["ev-1"]


class TestResearch:
    def test_request_defaults(self) -> None:
        request = StartAnalysisRequest(company_name="KANSBE", website="https://kansbe.com")
        assert request.competitor_count == 5
        assert request.research_depth == "standard"
        assert request.report_language == "en"

    def test_request_rejects_blank_company(self) -> None:
        with pytest.raises(ValidationError):
            StartAnalysisRequest(company_name="   ", website="https://a.com")

    def test_request_rejects_competitor_count_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            StartAnalysisRequest(company_name="A", website="https://a.com",
                                 competitor_count=0)

    def test_question_requires_a_known_category(self) -> None:
        with pytest.raises(ValidationError):
            ResearchQuestion(id="q1", category="astrology", question="?",
                             priority="high", status="pending")

    def test_question_accepts_known_category(self) -> None:
        q = ResearchQuestion(id="q1", category=AnalysisCategory.PRICING,
                             question="How is pricing structured?", priority="high",
                             status="pending")
        assert q.category is AnalysisCategory.PRICING

    def test_report_config_disables_graph_by_default(self) -> None:
        assert ReportConfig().enable_graph_rag is False

    def test_plan_starts_with_no_answers(self) -> None:
        plan = ResearchPlan(research_scope=["pricing"], research_questions=[])
        assert plan.research_questions == []
        assert plan.priority_levels == {}


class TestFindingInsightAndMarket:
    def test_finding_requires_at_least_one_evidence_link(self) -> None:
        with pytest.raises(ValidationError):
            Finding(id="f1", company="Acme", category="pricing", statement="",
                    observation="x", evidence_ids=[])

    def test_finding_keeps_interpretation_separate(self) -> None:
        finding = Finding(id="f1", company="Acme", category="pricing",
                          statement="Low entry price", observation="Lists $5/mo plan",
                          interpretation="Price-led positioning", evidence_ids=["ev-1"])
        assert finding.observation != finding.interpretation

    def test_insight_type_must_be_known(self) -> None:
        with pytest.raises(ValidationError):
            Insight(type="vibes", observation="x", evidence=["ev-1"])

    def test_market_insight_separates_evidence_from_opportunity(self) -> None:
        insight = MarketInsight(observation="Two of five competitors list no free tier",
                                interpretation="Entry pricing is a differentiator",
                                potential_opportunity="Offer a transparent free tier",
                                evidence=["ev-1", "ev-2"])
        assert len(insight.evidence) == 2


class TestComparisonAndReport:
    def test_comparison_cell_records_unknown_rather_than_blank(self) -> None:
        cell = ComparisonCell.unknown()
        assert cell.value == UNKNOWN_NOT_AVAILABLE
        assert cell.is_known is False

    def test_matrix_keeps_missing_values_unknown(self) -> None:
        matrix = ComparisonMatrix(
            companies=["Acme", "Beta"],
            rows=[{"field": "pricing", "values": {"Acme": None, "Beta": "$10/mo"}}],
        )
        assert matrix.rows[0]["values"]["Acme"] == UNKNOWN_NOT_AVAILABLE

    def test_report_metadata_requires_timestamps(self) -> None:
        with pytest.raises(ValidationError):
            ReportMetadata(run_id="r1", company="Acme", website="https://a.com",
                           industry="Freelancing", country="India",
                           research_started_at="", research_completed_at="")

    def test_report_bundle_round_trips(self) -> None:
        bundle = ReportBundle(
            metadata=ReportMetadata(run_id="r1", company="Acme", website="https://a.com",
                                    industry="Freelancing", country="India",
                                    research_started_at="2026-01-01T00:00:00Z",
                                    research_completed_at="2026-01-01T01:00:00Z"),
            sources=[_source()],
        )
        assert ReportBundle.model_validate_json(bundle.model_dump_json()).metadata.run_id == "r1"

    def test_report_bundle_rejects_extra_top_level_keys(self) -> None:
        with pytest.raises(ValidationError):
            ReportBundle(
                metadata=ReportMetadata(run_id="r1", company="A", website="https://a.com",
                                        industry="i", country="c",
                                        research_started_at="2026-01-01T00:00:00Z",
                                        research_completed_at="2026-01-01T01:00:00Z"),
                surprise="should not be allowed",
            )


class TestPageTypeIsClassifiable:
    def test_page_type_is_a_str_enum_for_json(self) -> None:
        assert PageType.PRICING.value == "pricing"
        assert InsightType.PRICING.value == "pricing"
