"""Marketing analysis: the story the company tells, kept apart from the truth.

This schema is the clearest case for separating what was *observed* from what
was *inferred*, and it has separate fields for each: ``observations`` and
``inferences``. That split is the point of the prompt. Marketing analysis is
where a model is most tempted to conclude — the copy implies a strategy, and the
model will state the strategy as though the company had — so the instruction is
to describe the messaging as messaging and to keep any conclusion about
strategy in the inferences field, where it will not be read as something the
company published.
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.schemas.marketing import MarketingAnalysis
from app.schemas.provider import ChatMessage

TASK_TYPE = "marketing_analysis"
OUTPUT_SCHEMA = MarketingAnalysis

SECTIONS = [
    "The brand messaging as published: the headlines, value propositions and "
    "themes the company's own pages use. Report these as the company's claims, "
    "in its framing.",
    "The content themes, calls to action, case studies and lead magnets the "
    "pages actually contain.",
    "Observations — what is plainly there. Inferences — what you conclude from "
    "it. Keep these strictly apart. An inference written as an observation "
    "becomes a fabricated quote somewhere downstream, because downstream code "
    "cannot tell which field it came from.",
    "The channels evidenced by the retrieved pages. Do not list a channel "
    "because a company of this type would plausibly use it.",
    "What you could not establish, in the unknowns list.",
]

PURPOSE = (
    "Analyse this company's marketing, reporting published messaging as "
    "observations and any conclusion about strategy as inference."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"company": "Acme Cloud", "value_proposition": "observability without '
            'the enterprise price tag", "observations": ["pricing page leads with '
            '$10 per seat"], "inferences": ["price-led positioning"], '
            '"unknowns": ["paid channel mix"], "confidence": "low"}'
        ),
    )
