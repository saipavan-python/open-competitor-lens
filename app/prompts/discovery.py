"""Competitor discovery: proposing candidates, not endorsing them.

Discovery is deliberately wide and uncritical. The prompt is told to return
candidates it is only mildly confident about, because the validation step that
follows is where a weak candidate is supposed to die, and a discovery step that
pre-filters aggressively just hides the candidates a reader would have wanted to
see rejected. It carries no score for the same reason: an LLM's ranking is not
evidence of relevance, and a score in the output invites downstream code to
trust it.
"""

from __future__ import annotations

from pydantic import Field

from app.prompts.base import PromptContext, build_messages
from app.schemas.common import DomainModel
from app.schemas.competitor import CompetitorCandidate
from app.schemas.provider import ChatMessage

TASK_TYPE = "competitor_discovery"

SECTIONS = [
    "Companies that a buyer would plausibly weigh against this one: same buyer, "
    "same problem, substitutable product.",
    "For each candidate, the website you saw it at and the query that surfaced "
    "it, so the claim can be traced back to a retrieval rather than to memory.",
    "Include candidates you are only moderately confident about. A later step "
    "validates them, and a candidate rejected there is recorded as rejected — "
    "which is more useful than never surfacing it.",
    "Do not include the target company itself, its parent, or a reseller of it. "
    "Those are identity mistakes, not competitors.",
]


class CompetitorDiscovery(DomainModel):
    """The discovery step's output: candidates, none of them yet endorsed."""

    candidates: list[CompetitorCandidate] = Field(default_factory=list)


OUTPUT_SCHEMA = CompetitorDiscovery

PURPOSE = (
    "Propose companies that compete with this one. These are candidates for "
    "validation, not conclusions."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"candidates": [{"name": "Beta Metrics", "website": "https://beta.example", '
            '"description": "Seat-based monitoring", "industry": "observability", '
            '"discovery_query": "acme cloud monitoring alternatives"}]}'
        ),
    )
