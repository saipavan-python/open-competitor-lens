"""Picking the official domain is the decision the rest of the run hangs off.

Identity resolution, source authority, and the self-exclusion in competitor
validation are all keyed to this answer, so the tests below are mostly about
refusing to guess. A wrong domain does not fail loudly — it produces a confident,
well-cited report about a different company, which is the worst available
outcome.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.agents.base import AgentStatus
from app.agents.company_agent import CompanyAgent, name_similarity
from app.core.constants import PageType, RetrievalStatus, SourceType
from app.schemas.evidence import Source
from app.schemas.search import SearchHit


def _hit(url: str, *, title: str, snippet: str = "") -> SearchHit:
    return SearchHit(
        url=url,
        title=title,
        snippet=snippet,
        domain=None,
        rank=1,
        provider="test",
        retrieved_at=datetime.now(UTC),
    )


def _retrieved(
    domain: str, *, title: str, text: str = "real page body", source_type=SourceType.OFFICIAL_SITE
) -> Source:
    return Source(
        id=f"src-{domain}",
        url=f"https://{domain}/",
        canonical_url=f"https://{domain}/",
        title=title,
        text=text,
        domain=domain,
        source_type=source_type,
        page_type=PageType.HOMEPAGE,
        retrieval_status=RetrievalStatus.OK,
    )


class TestNameSimilarity:
    def test_identical_names_score_one(self) -> None:
        assert name_similarity("Acme Cloud", "Acme Cloud") == 1.0

    def test_a_legal_suffix_does_not_lower_the_score(self) -> None:
        """Otherwise every registered company fails to match itself."""
        assert name_similarity("Acme Cloud", "Acme Cloud Private Limited") == 1.0

    def test_case_and_spacing_are_irrelevant(self) -> None:
        assert name_similarity("ACME  Cloud", "acme cloud") == 1.0

    def test_a_shared_token_scores_partially(self) -> None:
        assert 0 < name_similarity("Acme", "Acme Cloud") < 1.0

    def test_unrelated_names_score_zero(self) -> None:
        assert name_similarity("Acme", "Globex") == 0.0

    def test_an_embedded_brand_still_counts(self) -> None:
        """A subsidiary or trading name that contains the brand."""
        assert name_similarity("Acme", "Acme Digital") > 0.0

    def test_empty_input_scores_zero(self) -> None:
        assert name_similarity("", "Acme") == 0.0


class TestResolveFromSearch:
    def test_the_own_domain_is_chosen(self) -> None:
        result = CompanyAgent().resolve(
            "Acme Cloud",
            [_hit("https://acmecloud.com/", title="Acme Cloud")],
        )
        assert result.domain == "acmecloud.com"
        assert result.status == str(AgentStatus.OK)
        assert result.confidence >= 0.75

    def test_the_chosen_candidate_is_ranked_first(self) -> None:
        result = CompanyAgent().resolve(
            "Acme Cloud",
            [
                _hit("https://acmedirectory.example/acme", title="Acme Cloud reviews"),
                _hit("https://acmecloud.com/", title="Acme Cloud"),
            ],
        )
        assert result.domain == "acmecloud.com"
        assert result.considered[0].domain == "acmecloud.com"

    def test_a_directory_page_loses_to_the_company_itself(self) -> None:
        """A page about a company is not the company."""
        result = CompanyAgent().resolve(
            "Acme Cloud",
            [
                _hit("https://g2.example/products/acme-cloud", title="Acme Cloud"),
                _hit("https://acmecloud.com/", title="Acme Cloud hosting"),
            ],
        )
        assert result.domain == "acmecloud.com"

    def test_ranking_is_deterministic(self) -> None:
        """Equally good candidates must resolve the same way on every run.

        Two TLDs for the same company name tie exactly, so only the tiebreak
        separates them; without it, which one the run profiles would depend on
        the order a search engine happened to return results in.
        """
        hits = [
            _hit("https://acmecloud.net/", title="Acme Cloud"),
            _hit("https://acmecloud.com/", title="Acme Cloud"),
        ]
        first = CompanyAgent().resolve("Acme Cloud", hits)
        second = CompanyAgent().resolve("Acme Cloud", list(reversed(hits)))
        assert first.domain == second.domain == "acmecloud.com"

    def test_a_weak_domain_is_refused_even_when_it_is_the_only_hit(self) -> None:
        """Determinism does not mean the run must pick something."""
        result = CompanyAgent().resolve(
            "Acme Cloud", [_hit("https://a.example/", title="Acme Cloud")]
        )
        assert result.domain is None
        assert result.status == str(AgentStatus.PARTIAL)


class TestRefusingToGuess:
    def test_no_hits_yields_a_partial_result_not_a_guess(self) -> None:
        result = CompanyAgent().resolve("Acme Cloud", [])
        assert result.domain is None
        assert result.status == str(AgentStatus.PARTIAL)
        assert result.rejected_reason

    def test_a_weak_match_is_refused_rather_than_profiled(self) -> None:
        """A plausible-looking wrong domain is worse than no domain."""
        result = CompanyAgent().resolve(
            "Acme Cloud",
            [_hit("https://globex.example/", title="Globex Corporation")],
        )
        assert result.domain is None
        assert result.status == str(AgentStatus.PARTIAL)
        assert "refusing" in (result.rejected_reason or "")

    def test_a_refused_resolution_still_shows_what_was_considered(self) -> None:
        """So a reader can see the judgement rather than just its absence."""
        result = CompanyAgent().resolve(
            "Acme Cloud", [_hit("https://globex.example/", title="Globex")]
        )
        assert [candidate.domain for candidate in result.considered] == ["globex.example"]

    def test_the_rejection_reason_names_the_score(self) -> None:
        result = CompanyAgent().resolve(
            "Acme Cloud", [_hit("https://globex.example/", title="Globex")]
        )
        assert "0." in (result.rejected_reason or "") or "below" in (result.rejected_reason or "")

    def test_unusable_hits_are_ignored(self) -> None:
        result = CompanyAgent().resolve(
            "Acme Cloud", [_hit("not a url", title="Acme Cloud")]
        )
        assert result.domain is None
        assert result.status == str(AgentStatus.PARTIAL)


class TestRetrievedPagesWin:
    def test_a_fetched_official_page_outranks_a_better_ranked_hit(self) -> None:
        """Authority is not a matter of opinion once the page has been fetched."""
        result = CompanyAgent().resolve(
            "Acme Cloud",
            [_hit("https://acmecloud.com/", title="Acme Cloud")],
            sources=[_retrieved("acmecloud.com", title="Acme Cloud")],
        )
        assert result.domain == "acmecloud.com"
        assert result.confidence == 1.0

    def test_a_retrieved_domain_wins_even_with_a_mismatched_title(self) -> None:
        """Companies whose homepage title is not their name still resolve."""
        result = CompanyAgent().resolve(
            "Acme Cloud",
            [_hit("https://acmecloud.com/", title="Welcome")],
            sources=[_retrieved("acmecloud.com", title="Fast, simple hosting")],
        )
        assert result.domain == "acmecloud.com"

    def test_a_retrieved_page_with_no_text_is_not_an_anchor(self) -> None:
        """A snippet is not a page, however good its title looks."""
        source = _retrieved("acmecloud.com", title="Acme Cloud")
        source.text = None
        result = CompanyAgent().resolve(
            "Acme Cloud", [], sources=[source]
        )
        assert result.domain is None

    def test_an_unrelated_retrieved_page_is_ignored(self) -> None:
        result = CompanyAgent().resolve(
            "Acme Cloud",
            [],
            sources=[_retrieved("globex.example", title="Globex Corporation")],
        )
        assert result.domain is None
        assert result.status == str(AgentStatus.PARTIAL)


class TestIdentityOutput:
    def test_the_resolution_carries_an_identity(self) -> None:
        result = CompanyAgent().resolve(
            "Acme Cloud", [_hit("https://acmecloud.com/", title="Acme Cloud")]
        )
        assert result.identity is not None
        assert result.identity.canonical_name == "Acme Cloud"
        assert "acmecloud.com" in (result.identity.domains or [])

    def test_a_failed_resolution_carries_no_identity(self) -> None:
        result = CompanyAgent().resolve("Acme Cloud", [])
        assert result.identity is None
