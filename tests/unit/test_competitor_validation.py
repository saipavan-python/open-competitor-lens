"""Competitor validation: the narrowest gate in the system.

Discovery is meant to be generous — it returns candidates it is only mildly
confident about, on the understanding that this step will throw most of them
away. So these tests are mostly about what gets *rejected*, because the failure
that matters is the permissive one: a company that does not compete, admitted on
the strength of a search snippet, turns into a row in the comparison table that
looks exactly as well-sourced as a row built from a vendor's own pricing page.

The other thing worth guarding is the pool size. Validation is supposed to be
lossy, so a generous pool is what makes "exactly N competitors" achievable
without the validation bar being dragged down to fit whatever survived.
"""

from __future__ import annotations

import pytest

from app.agents.competitor_agent import (
    CandidateSignals,
    CompetitorAgent,
    Validator,
)
from app.core.constants import SourceType
from app.schemas.competitor import CompetitorCandidate, ValidationSignal
from app.schemas.evidence import Source
from app.utils.normalization import company_identity_key

#: A source rich enough to back a "these two compete" claim.
PRIMARY = Source(
    id="src-primary",
    url="https://g2.com/compare/acme-vs-beta",
    title="Acme vs Beta",
    source_type=SourceType.REVIEW_PLATFORM,
    company="Beta",
)
#: A directory entry written by a third party about somebody else.
SECONDARY = Source(
    id="src-secondary",
    url="https://directory.example/beta",
    source_type=SourceType.DIRECTORY,
    company="Beta",
)
#: What a search engine returns without anyone retrieving the page behind it.
SNIPPET = Source(
    id="src-snippet",
    url="https://search.example/?q=acme+alternatives",
    source_type=SourceType.SEARCH_SNIPPET,
    company="Beta",
)

TARGET_KEY = company_identity_key("Acme Cloud")


def _candidate(name: str = "Beta Metrics") -> CompetitorCandidate:
    return CompetitorCandidate(
        name=name, website=f"https://{name.split()[0].lower()}.example"
    )


def _signal(name: str, source_ids: list[str]) -> ValidationSignal:
    return ValidationSignal(signal=name, supported=True, source_ids=source_ids)


def _validator(*sources: Source) -> Validator:
    return Validator(
        target_identity_key=TARGET_KEY,
        sources={source.id: source for source in sources},
    )


def _strong_signals() -> list[ValidationSignal]:
    return [
        _signal("same_target_segment", ["src-primary"]),
        _signal("listed_as_alternative", ["src-primary"]),
    ]


class TestSelfIdentification:
    def test_a_candidate_that_is_the_target_is_dropped(self) -> None:
        outcome = _validator(PRIMARY).validate(
            CandidateSignals(
                candidate=_candidate("Acme Cloud"), signals=_strong_signals()
            )
        )
        assert not outcome.accepted
        assert outcome.reason == "self"

    def test_the_target_is_dropped_even_with_strong_signals(self) -> None:
        """Identity is checked first: nothing outranks 'it is the company itself'."""
        outcome = _validator(PRIMARY).validate(
            CandidateSignals(
                candidate=_candidate("ACME  CLOUD"), signals=_strong_signals()
            )
        )
        assert outcome.reason == "self"

    def test_a_trading_style_variant_of_the_target_is_dropped(self) -> None:
        """Case and spacing are not a different company."""
        outcome = _validator(PRIMARY).validate(
            CandidateSignals(
                candidate=_candidate("Acme Cloud, Inc."), signals=_strong_signals()
            )
        )
        assert not outcome.accepted
        assert outcome.reason == "self"

    def test_an_alias_of_the_target_is_also_dropped(self) -> None:
        """A parent or trading name is the same company, not a competitor of it."""
        outcome = _validator(PRIMARY).validate(
            CandidateSignals(
                candidate=_candidate("Acme Cloud Private Limited"),
                signals=_strong_signals(),
            )
        )
        assert not outcome.accepted
        assert outcome.reason == "self"

    def test_a_similarly_named_but_distinct_company_is_not_dropped(self) -> None:
        """The merge rule is exact normalised equality, not substring matching."""
        outcome = _validator(PRIMARY).validate(
            CandidateSignals(
                candidate=_candidate("Acme Digital"), signals=_strong_signals()
            )
        )
        assert outcome.accepted


class TestEvidenceBacking:
    def test_a_candidate_backed_only_by_a_search_snippet_is_rejected(self) -> None:
        outcome = _validator(SNIPPET).validate(
            CandidateSignals(
                candidate=_candidate(), signals=[_signal("listed_as_alternative", ["src-snippet"])]
            )
        )
        assert not outcome.accepted
        assert "snippet" in outcome.reason.lower()

    def test_a_snippet_backed_candidate_is_rejected_even_with_many_signals(self) -> None:
        """Volume is not authority: five snippet-backed signals are still a snippet."""
        outcome = _validator(SNIPPET).validate(
            CandidateSignals(
                candidate=_candidate(),
                signals=[
                    _signal("same_target_segment", ["src-snippet"]),
                    _signal("product_overlap", ["src-snippet"]),
                    _signal("co_occurrence", ["src-snippet"]),
                    _signal("listed_as_alternative", ["src-snippet"]),
                    _signal("substitutable", ["src-snippet"]),
                ],
            )
        )
        assert not outcome.accepted

    def test_one_primary_backed_signal_is_enough(self) -> None:
        outcome = _validator(PRIMARY).validate(
            CandidateSignals(
                candidate=_candidate(), signals=[_signal("product_overlap", ["src-primary"])]
            )
        )
        assert outcome.accepted

    def test_a_signal_citing_an_unretrieved_source_counts_for_nothing(self) -> None:
        """A source id the run never fetched is exactly the hallucination this
        system exists to stop, and it must not become authority here either."""
        outcome = _validator(PRIMARY).validate(
            CandidateSignals(
                candidate=_candidate(),
                signals=[_signal("product_overlap", ["src-never-fetched"])],
            )
        )
        assert not outcome.accepted

    def test_a_candidate_with_no_signals_at_all_is_rejected(self) -> None:
        outcome = _validator(PRIMARY).validate(
            CandidateSignals(candidate=_candidate(), signals=[])
        )
        assert not outcome.accepted

    def test_an_unsupported_signal_does_not_count(self) -> None:
        outcome = _validator(PRIMARY).validate(
            CandidateSignals(
                candidate=_candidate(),
                signals=[
                    ValidationSignal(
                        signal="product_overlap", supported=False, source_ids=["src-primary"]
                    )
                ],
            )
        )
        assert not outcome.accepted


class TestPoolSelection:
    def _agent(self) -> CompetitorAgent:
        return CompetitorAgent(
            validator=_validator(PRIMARY, SECONDARY, SNIPPET)
        )

    def _pool(self, size: int) -> list[CandidateSignals]:
        return [
            CandidateSignals(
                candidate=_candidate(f"Firm{index}"),
                signals=[
                    _signal("product_overlap", ["src-primary"]),
                    _signal("listed_as_alternative", ["src-primary"]),
                ],
            )
            for index in range(size)
        ]

    def test_ten_candidates_and_a_target_of_five_yields_exactly_five(self) -> None:
        result = self._agent().select(self._pool(10), competitor_count=5)
        assert len(result.competitors) == 5

    def test_the_pool_is_wider_than_the_target(self) -> None:
        """Validation must be lossy, or the target count is unreachable honestly."""
        assert self._agent().pool_size_for(competitor_count=5) > 5

    def test_the_pool_stays_within_the_specified_multiple(self) -> None:
        agent = self._agent()
        for count in (3, 5, 8):
            size = agent.pool_size_for(competitor_count=count)
            assert 2 * count <= size <= 4 * count, count

    def test_rejected_candidates_are_recorded_with_their_reason(self) -> None:
        pool = self._pool(4)
        pool.append(
            CandidateSignals(
                candidate=_candidate("Acme Cloud"),
                signals=[_signal("product_overlap", ["src-primary"])],
            )
        )
        pool.append(
            CandidateSignals(
                candidate=_candidate("Snippet Firm"),
                signals=[_signal("product_overlap", ["src-snippet"])],
            )
        )
        result = self._agent().select(pool, competitor_count=2)
        assert len(result.competitors) == 2
        reasons = {outcome.candidate.name: outcome.reason for outcome in result.rejected}
        assert reasons["Acme Cloud"] == "self"
        assert "snippet" in reasons["Snippet Firm"].lower()

    def test_a_pool_too_small_to_fill_the_target_is_reported_short(self) -> None:
        """Returning three competitors when five were asked for must be visible."""
        result = self._agent().select(self._pool(3), competitor_count=5)
        assert len(result.competitors) == 3
        assert result.status == "partial"

    def test_an_exact_fit_is_a_complete_run(self) -> None:
        result = self._agent().select(self._pool(5), competitor_count=5)
        assert result.status == "ok"


class TestRelevanceReason:
    def test_an_accepted_competitor_states_why_it_is_in_the_comparison(self) -> None:
        result = self._agent_result()
        assert result.competitors[0].relevance_reason
        assert "product_overlap" in result.competitors[0].relevance_reason

    def _agent_result(self):  # noqa: ANN202 - small helper
        agent = CompetitorAgent(validator=_validator(PRIMARY))
        pool = [
            CandidateSignals(
                candidate=_candidate(),
                signals=[
                    _signal("product_overlap", ["src-primary"]),
                    _signal("co_occurrence", ["src-primary"]),
                ],
            )
        ]
        return agent.select(pool, competitor_count=1)


class TestAgentContract:
    def test_the_agent_declares_its_contract(self) -> None:
        agent = CompetitorAgent(validator=_validator(PRIMARY))
        assert agent.name
        assert agent.goal
        assert agent.timeout > 0
        assert agent.max_iterations >= 1

    def test_the_agent_reports_its_output_schema(self) -> None:
        from app.schemas.competitor import ValidatedCompetitor

        assert CompetitorAgent(validator=_validator(PRIMARY)).output_schema is (
            ValidatedCompetitor
        )


class TestValidatorStandalone:
    def test_a_validator_needs_no_llm_or_network(self) -> None:
        """Validation is the step that must not fail because a provider is down."""
        assert _validator(PRIMARY).validate(
            CandidateSignals(candidate=_candidate(), signals=_strong_signals())
        ).accepted

    def test_the_primary_signal_count_is_reported(self) -> None:
        """A snippet-backed signal is counted apart from a retrieved one."""
        outcome = _validator(PRIMARY, SNIPPET).validate(
            CandidateSignals(
                candidate=_candidate(),
                signals=[
                    _signal("product_overlap", ["src-primary"]),
                    _signal("co_occurrence", ["src-snippet"]),
                ],
            )
        )
        assert outcome.primary_signals == 1
        assert outcome.accepted


@pytest.mark.parametrize("name", ["", "   "])
def test_a_blank_candidate_name_is_impossible(name: str) -> None:
    """The schema already refuses it; this documents that it is not our job."""
    with pytest.raises(ValueError):
        CompetitorCandidate(name=name)
