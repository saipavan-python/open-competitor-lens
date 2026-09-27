"""Research planning: what to ask, and which source tier can answer it.

The planner runs before anything is retrieved, so it has no evidence to reason
from and must not pretend otherwise. It is therefore the one prompt whose output
is *not* gated by the evidence service — it produces questions, not claims.
What it must get right is coverage: a plan that omits a category produces a
report that is silent about it, and silence reads as "nothing to report" rather
than "we did not look".
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.schemas.provider import ChatMessage
from app.schemas.research import ResearchPlan

TASK_TYPE = "research_planning"
OUTPUT_SCHEMA = ResearchPlan

SECTIONS = [
    "The questions to answer, each scoped to one category and phrased so that a "
    "specific page could answer it.",
    "For each question, the source tier required to answer it — an official "
    "pricing page, a review platform, documentation, and so on. Pricing claims "
    "require the vendor's own page; a blog quoting a price is not a substitute.",
    "The order of work, highest value first, so a run that exhausts its budget "
    "still answers the questions that matter.",
    "Any category you are deliberately excluding, and why. A declared exclusion "
    "is a decision; a silent gap is a bug.",
]

PURPOSE = (
    "Write the research plan for this competitor analysis: the questions to ask, "
    "the source tier each requires, and the order to pursue them."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"research_scope": ["pricing", "positioning"], '
            '"research_questions": [{"id": "q1", "category": "pricing", '
            '"question": "What does the vendor charge per seat per month?", '
            '"priority": "high", "required_sources": ["official_site"]}]}'
        ),
    )
