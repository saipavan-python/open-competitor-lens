"""Query generation: turning a question into searches that could answer it.

Query generation runs before retrieval, so it is the one prompt whose output is
judged on reach rather than on support — there is no evidence yet to check it
against. The failure that matters is narrow queries: asking one precise question
per fact tends to return the vendor's own marketing page and nothing else, and
the whole point of searching is to find the pages a company did not write about
itself.

So the prompt asks for a spread of query shapes — the vendor's own wording, a
third party's wording, comparison and complaint framings — and asks for the
expected source tier so the retrieval step knows what a useful hit looks like.
"""

from __future__ import annotations

from pydantic import Field

from app.prompts.base import PromptContext, build_messages
from app.schemas.common import DomainModel
from app.schemas.provider import ChatMessage
from app.schemas.research import SearchStrategy

TASK_TYPE = "query_generation"

SECTIONS = [
    "Queries aimed at the vendor's own pages: pricing, documentation, plans, "
    "terms. These are the queries most likely to return an authoritative page, "
    "and the ones worth running first.",
    "Queries aimed at what the vendor does not say about itself: comparisons, "
    "alternatives, reviews, complaints, migration stories. A search that only "
    "finds the vendor's marketing finds nothing a competitor has not already "
    "published about itself.",
    "Queries framed the way a buyer would phrase the problem, not the way the "
    "vendor names their product — buyers search for the problem, and matching "
    "the vendor's vocabulary narrows the results to the vendor.",
    "For each query, the source tier a useful result would come from, so the "
    "retrieval step can tell a real hit from a restatement.",
    "Do not include the target company's name in a query intended to find its "
    "critics; name-filtered queries retrieve only praise.",
]


class QuerySet(DomainModel):
    """The queries generated for one research question."""

    queries: list[str] = Field(default_factory=list)
    expected_source_types: list[str] = Field(default_factory=list)
    strategy: SearchStrategy = Field(default_factory=SearchStrategy)


OUTPUT_SCHEMA = QuerySet

PURPOSE = (
    "Generate the search queries most likely to answer this question, covering "
    "the vendor's own pages and the pages it did not write."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"queries": ["acme cloud pricing per seat", '
            '"acme cloud vs alternatives", "acme cloud reviews complaints", '
            '"seat based monitoring tools for platform teams"], '
            '"expected_source_types": ["official_site", "review_platform"]}'
        ),
    )
