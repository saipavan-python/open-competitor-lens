"""Product analysis: the offering, and what actually distinguishes it.

"Differentiators" is the field most likely to be fabricated, because a model
asked "how is this different" will produce a differentiation for any product it
has heard of. So the prompt requires each differentiator to be traceable to
something the retrieved pages actually say the company does, and treats a
feature every competitor also has as a non-differentiator — which is usually the
honest answer.
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.schemas.products import ProductAnalysis
from app.schemas.provider import ChatMessage

TASK_TYPE = "product_analysis"
OUTPUT_SCHEMA = ProductAnalysis

SECTIONS = [
    "The products and services on offer, named as the company names them.",
    "The features documented as available today, separated from anything the "
    "pages describe as planned or in beta.",
    "The segments and use cases the company states it serves.",
    "The differentiators — and be strict about what qualifies. A capability "
    "every competitor in this market also has is not a differentiator. If you "
    "cannot point to something the retrieved pages present as distinctive, the "
    "honest entry is an empty list, not a plausible-sounding one.",
    "The positioning, left at its not-public sentinel if the pages do not state "
    "one clearly.",
]

PURPOSE = (
    "Analyse this company's product offering from the retrieved pages, including "
    "only differentiators the evidence actually supports."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"company": "Acme Cloud", "products": ["Cloud monitoring"], '
            '"features": ["SSO", "audit logs"], '
            '"differentiators": ["per-seat pricing with a free tier"], '
            '"positioning": "not_public", "confidence": "medium"}'
        ),
    )
