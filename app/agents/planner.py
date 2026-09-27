"""Research planning.

The plan drives every retrieval that follows, so it is worth being precise about
what it decides. For each analysis category it fixes three things: the question
to ask, the *tier* of source that can answer it, and the priority. The tier is
the part that does real work — it tells the retrieval step what a useful hit
looks like before anything has been fetched, and it is why a pricing question is
never satisfied by a blog that quotes a price.

**Planning is deterministic.** A rule-based plan over a fixed taxonomy is
reproducible, free, and identical on a re-run, and a re-run that produces a
different plan makes a diff of two runs almost impossible to read. The
``research_planning`` prompt exists for the case where the goal is phrased in a
way the keyword table cannot resolve, and the escalation is explicit rather than
implicit: a goal that matches nothing is treated as "cover everything", and the
resulting full-coverage plan is reported as such.

Tailoring cuts both ways and both failures are guarded by tests. Emitting SEO
and social questions for a request that was "compare pricing only" spends the
run's whole page budget on questions nobody asked. Omitting a category produces a
report that is silent about it, and silence reads as "nothing to report" rather
than "we did not look" — so a tailored plan records what it excluded and why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.agents.base import Agent, AgentContext, AgentResult, AgentStatus
from app.core.constants import AnalysisCategory, SourceType
from app.core.logging import get_logger
from app.schemas.common import DomainModel
from app.schemas.research import ResearchPlan, ResearchQuestion, SearchStrategy

log = get_logger(__name__)

#: One question per analysis category, and no more. A plan that cannot be
#: expressed in this many questions cannot be funded by a per-run page budget.
MAX_QUESTIONS = len(AnalysisCategory)


@dataclass(frozen=True)
class CategorySpec:
    """How to research one analysis category."""

    question: str
    required_sources: tuple[str, ...]
    priority: str
    keywords: tuple[str, ...]


#: The taxonomy, ordered by priority. Order matters twice over: it is the
#: emission order for a full-coverage plan, and it is why a run that exhausts
#: its budget still answers the questions a reader cares about most.
CATEGORY_SPECS: dict[AnalysisCategory, CategorySpec] = {
    AnalysisCategory.PRICING: CategorySpec(
        question="What does this company charge, on what terms, and for which plans?",
        required_sources=(SourceType.OFFICIAL_SITE.value, SourceType.OFFICIAL_DOCS.value),
        priority="high",
        keywords=("pricing", "price", "cost", "costs", "plan", "plans", "cheap",
                  "cheapest", "per seat", "subscription", "quote"),
    ),
    AnalysisCategory.PRODUCT: CategorySpec(
        question="What does this company actually sell, and to whom?",
        required_sources=(SourceType.OFFICIAL_SITE.value, SourceType.OFFICIAL_DOCS.value),
        priority="high",
        keywords=("product", "products", "feature", "features", "offering",
                  "what does it do", "capabilities", "use case"),
    ),
    AnalysisCategory.WEBSITE: CategorySpec(
        question="What does the company say about itself on its own site?",
        required_sources=(SourceType.OFFICIAL_SITE.value,),
        priority="high",
        keywords=("website", "site", "homepage", "about", "company profile",
                  "who are they", "overview"),
    ),
    AnalysisCategory.MARKETING: CategorySpec(
        question="How does the company position and market itself?",
        required_sources=(SourceType.OFFICIAL_SITE.value,),
        priority="medium",
        keywords=("marketing", "messaging", "brand", "positioning", "campaign",
                  "advertising", "pitch"),
    ),
    AnalysisCategory.REVIEW: CategorySpec(
        question="What do customers say about this company, good and bad?",
        required_sources=(SourceType.REVIEW_PLATFORM.value,),
        priority="medium",
        keywords=("review", "reviews", "complaint", "complaints", "customer",
                  "customers", "rating", "ratings", "testimonial", "hate", "love"),
    ),
    AnalysisCategory.TECHNOLOGY: CategorySpec(
        question="What technology does this company appear to run on?",
        required_sources=(SourceType.OFFICIAL_SITE.value, SourceType.PRIMARY_PUBLIC.value),
        priority="medium",
        keywords=("technology", "tech", "stack", "engineering", "infrastructure",
                  "architecture", "hosting"),
    ),
    AnalysisCategory.MARKET: CategorySpec(
        question="What market does this company operate in, and how is it moving?",
        required_sources=(SourceType.NEWS.value, SourceType.INDEPENDENT.value,
                          SourceType.REVIEW_PLATFORM.value),
        priority="medium",
        keywords=("market", "industry", "sector", "trend", "trends", "segment",
                  "landscape", "size"),
    ),
    AnalysisCategory.SEO: CategorySpec(
        question="How is this company configured for search?",
        required_sources=(SourceType.OFFICIAL_SITE.value,),
        priority="low",
        keywords=("seo", "search engine", "ranking", "rankings", "organic",
                  "indexed", "visibility"),
    ),
    AnalysisCategory.SOCIAL: CategorySpec(
        question="What presence does this company have on social platforms?",
        required_sources=(SourceType.OFFICIAL_SITE.value, SourceType.DIRECTORY.value),
        priority="low",
        keywords=("social", "twitter", "linkedin", "reddit", "github", "community",
                  "following", "presence"),
    ),
}

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


class PlannerInput(DomainModel):
    """What the planner needs to know. Declared so the contract is inspectable."""

    company: str
    research_goal: str = ""
    competitor_count: int = 3


class PlannerAgent(Agent):
    """Turns a research goal into an ordered, budgetable set of questions."""

    name = "planner"
    goal = "Decide what to research, in what order, and from which source tier."
    input_schema = PlannerInput
    output_schema = ResearchPlan
    task_type = "research_planning"

    # -- the plan ---------------------------------------------------------

    def build_plan(self, company: str, research_goal: str = "") -> ResearchPlan:
        """The deterministic core, usable without any provider."""
        included, excluded = self._select_categories(research_goal)
        ordered = sorted(included, key=lambda c: _PRIORITY_ORDER[CATEGORY_SPECS[c].priority])

        questions = [
            ResearchQuestion(
                id=f"q{position}",
                category=category,
                question=CATEGORY_SPECS[category].question,
                priority=CATEGORY_SPECS[category].priority,
                required_sources=list(CATEGORY_SPECS[category].required_sources),
            )
            for position, category in enumerate(ordered, start=1)
        ][:MAX_QUESTIONS]

        scope = [question.question for question in questions]
        return ResearchPlan(
            research_scope=scope,
            research_questions=questions,
            search_strategy=SearchStrategy(
                queries_per_competitor=6,
                preferred_source_types=sorted(
                    {
                        source
                        for question in questions
                        for source in question.required_sources
                    }
                ),
                use_browser_fallback=True,
            ),
            required_sources=sorted(
                {
                    source
                    for question in questions
                    for source in question.required_sources
                }
            ),
            priority_levels={
                question.id: question.priority for question in questions
            },
            excluded_categories=excluded,
        )

    def _select_categories(
        self, research_goal: str
    ) -> tuple[list[AnalysisCategory], list[AnalysisCategory]]:
        """Which categories to research, and which to declare as excluded.

        A goal that names nothing recognisable is treated as full coverage. The
        alternative — refusing to plan — hands the caller an empty plan, and an
        empty plan is the one outcome that cannot be improved on by a re-run.
        """
        goal = (research_goal or "").lower()
        if not goal.strip():
            return list(AnalysisCategory), []

        matched = {
            category
            for category, spec in CATEGORY_SPECS.items()
            if any(_mentions(goal, keyword) for keyword in spec.keywords)
        }
        if not matched:
            log.info("research goal matched no category; planning full coverage")
            return list(AnalysisCategory), []

        excluded = [category for category in AnalysisCategory if category not in matched]
        log.info(
            "research plan tailored",
            extra={"matched": sorted(c.value for c in matched),
                   "excluded": sorted(c.value for c in excluded)},
        )
        return list(matched), excluded

    # -- the agent contract -----------------------------------------------

    async def run(self, ctx: AgentContext) -> AgentResult:
        return self.run_sync(company=ctx.company, research_goal=ctx.research_goal)

    def run_sync(
        self, *, company: str = "", research_goal: str = ""
    ) -> AgentResult:
        """Synchronous entry point.

        Planning does no I/O, so there is nothing to await; this exists so the
        planner can be used from synchronous code and tested without an event
        loop, not to provide a second implementation.
        """
        plan = self.build_plan(company, research_goal)
        return self._result(status=AgentStatus.OK, payload=plan)


def _mentions(goal: str, keyword: str) -> bool:
    """Whether ``goal`` talks about ``keyword``.

    Word-boundary matched where the keyword is a single word, so "social" does
    not fire on "association" and "market" does not fire on "marketing".
    Multi-word keywords are substring matched, since a phrase like "per seat"
    has no useful word boundary at its edges.
    """
    if " " in keyword:
        return keyword in goal
    return re.search(rf"\b{re.escape(keyword)}\b", goal) is not None
