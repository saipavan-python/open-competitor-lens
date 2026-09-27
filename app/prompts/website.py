"""Website analysis: what the vendor's own site says about itself.

The official site is the most authoritative source available and the easiest to
over-read. Marketing copy is written to describe the product's best case, and a
model asked for a feature list will happily return the aspirational one. So the
prompt asks what is actually published, treats the site's own hedges ("planned",
"beta", "coming soon") as facts worth recording, and pushes everything it cannot
confirm into the unknowns list rather than dropping it.
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.schemas.company import CompanyProfile
from app.schemas.provider import ChatMessage

TASK_TYPE = "website_analysis"
OUTPUT_SCHEMA = CompanyProfile

SECTIONS = [
    "What the company says it sells: products, services, and the segments it "
    "names. Quote the site, not your summary of it.",
    "The stated value proposition, in the company's own framing where possible. "
    "If the site makes no clear claim, leave it unknown — a paraphrase you "
    "invented is worse than an empty field.",
    "Anything the site presents as planned, beta or forthcoming, recorded as "
    "such. A roadmap item is not a shipping feature and the difference is the "
    "whole value of this step.",
    "Everything you could not establish from the site, in the unknowns list. A "
    "field you filled from general knowledge about this company is a defect.",
]

PURPOSE = (
    "Describe this company from its own website, distinguishing what it ships "
    "today from what it says it plans to."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"name": "Acme Cloud", "products": ["Cloud monitoring"], '
            '"target_segments": ["platform engineering"], '
            '"unknowns": ["employee count"], "confidence": "medium"}'
        ),
    )
