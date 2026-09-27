"""Research planning: coverage, and tailoring to what was actually asked for.

A plan drives every retrieval that follows, so its two failure modes are
symmetrical and both are tested here. A plan that asks about the wrong things —
covering SEO and social presence when the request was "compare pricing only" —
spends the run's entire page budget on questions nobody asked. A plan that omits
a category produces a report that is silent about it, and silence reads as
"nothing to report" rather than "we did not look".

The source tier attached to each question is the other half. A pricing question
answered from a blog is not answered, and the tier is what tells the retrieval
step what a useful hit looks like before anything has been fetched.
"""

from __future__ import annotations

import pytest

from app.agents.planner import MAX_QUESTIONS, PlannerAgent
from app.core.constants import AnalysisCategory
from app.schemas.research import ResearchPlan

PRICING_ONLY = "compare pricing only"


@pytest.fixture()
def planner() -> PlannerAgent:
    return PlannerAgent()


def _run(planner: PlannerAgent, goal: str = "") -> ResearchPlan:
    """Run the planner with a stub LLM that echoes the plan it was asked for."""
    return planner.run_sync(research_goal=goal).payload


class TestTailoring:
    def test_a_pricing_only_goal_emits_no_seo_question(self, planner: PlannerAgent) -> None:
        plan = _run(planner, PRICING_ONLY)
        assert AnalysisCategory.SEO not in {
            question.category for question in plan.research_questions
        }

    def test_a_pricing_only_goal_still_asks_about_pricing(self, planner: PlannerAgent) -> None:
        plan = _run(planner, PRICING_ONLY)
        assert AnalysisCategory.PRICING in {
            question.category for question in plan.research_questions
        }

    def test_a_pricing_only_goal_excludes_unrelated_categories(
        self, planner: PlannerAgent
    ) -> None:
        plan = _run(planner, PRICING_ONLY)
        categories = {question.category for question in plan.research_questions}
        assert not categories & {AnalysisCategory.SEO, AnalysisCategory.SOCIAL}

    def test_the_exclusions_are_recorded_not_silently_dropped(
        self, planner: PlannerAgent
    ) -> None:
        """A declared exclusion is a decision; a silent gap is a bug."""
        plan = _run(planner, PRICING_ONLY)
        assert AnalysisCategory.SEO in plan.excluded_categories

    def test_an_empty_goal_covers_everything(self, planner: PlannerAgent) -> None:
        plan = _run(planner, "")
        assert len(plan.research_questions) == MAX_QUESTIONS

    def test_a_product_goal_includes_product_and_not_seo(self, planner: PlannerAgent) -> None:
        plan = _run(planner, "what does the product actually do")
        categories = {question.category for question in plan.research_questions}
        assert AnalysisCategory.PRODUCT in categories
        assert AnalysisCategory.SEO not in categories

    def test_a_reviews_goal_includes_the_review_category(self, planner: PlannerAgent) -> None:
        plan = _run(planner, "what do customers complain about")
        assert AnalysisCategory.REVIEW in {
            question.category for question in plan.research_questions
        }


class TestCoverage:
    def test_the_plan_never_exceeds_the_question_budget(self, planner: PlannerAgent) -> None:
        for goal in ("", "everything", "compare pricing only", "seo audit"):
            assert len(_run(planner, goal).research_questions) <= MAX_QUESTIONS

    def test_every_question_has_an_id(self, planner: PlannerAgent) -> None:
        plan = _run(planner, "")
        ids = [question.id for question in plan.research_questions]
        assert all(ids) and len(set(ids)) == len(ids)

    def test_every_question_declares_the_source_tier_it_needs(
        self, planner: PlannerAgent
    ) -> None:
        plan = _run(planner, "")
        assert all(question.required_sources for question in plan.research_questions)

    def test_a_pricing_question_requires_a_primary_source(self, planner: PlannerAgent) -> None:
        """A price quoted by a blog is not a price; the tier has to say so."""
        plan = _run(planner, "")
        pricing = next(
            q for q in plan.research_questions if q.category is AnalysisCategory.PRICING
        )
        assert "official_site" in pricing.required_sources

    def test_questions_are_ordered_by_priority(self, planner: PlannerAgent) -> None:
        """A run that exhausts its budget must still answer what matters."""
        plan = _run(planner, "")
        order = {"high": 0, "medium": 1, "low": 2}
        ranks = [order[q.priority] for q in plan.research_questions]
        assert ranks == sorted(ranks)

    def test_the_search_strategy_is_populated(self, planner: PlannerAgent) -> None:
        plan = _run(planner, "")
        assert plan.search_strategy.queries_per_competitor >= 1


class TestResultContract:
    def test_the_result_carries_the_plan_as_its_payload(self, planner: PlannerAgent) -> None:
        result = planner.run_sync(research_goal="")
        assert isinstance(result.payload, ResearchPlan)

    def test_a_plan_needs_no_credentials(self, planner: PlannerAgent) -> None:
        """Planning is deterministic, so it must work with no provider configured."""
        assert planner.run_sync(research_goal="").status == "ok"

    def test_the_agent_declares_its_contract(self, planner: PlannerAgent) -> None:
        assert planner.name
        assert planner.goal
        assert planner.output_schema is ResearchPlan
        assert planner.input_schema is not None

    def test_the_planner_uses_the_planning_prompt(self, planner: PlannerAgent) -> None:
        """The plan must come from the registered prompt, not from inline prose."""
        from app.prompts import get

        assert get(planner.task_type) is not None
