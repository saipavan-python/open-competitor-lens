"""Pricing analysis: the numbers a buyer will actually be quoted.

Pricing is the category where a wrong figure does the most damage, because it is
the one number a reader is likely to act on or quote back. Two failure modes
matter here specifically. A price from a third party — a blog, a review site, a
cached comparison table — is routinely out of date, and the current price is the
one the vendor publishes. And "starts at" is not a price: a plan table with
annual and monthly rates, a seat minimum and a usage tier has several prices,
and reporting the lowest as *the* price misstates the offer.

So the prompt insists on the vendor's own page, on the conditions attached to
each figure, and on the not-public sentinel when the vendor does not publish
one. Several vendors publish no prices at all, and recording that accurately is
itself a finding.
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.schemas.pricing import PricingAnalysis
from app.schemas.provider import ChatMessage

TASK_TYPE = "pricing_analysis"
OUTPUT_SCHEMA = PricingAnalysis

SECTIONS = [
    "Each published plan, with its price, currency and billing period exactly as "
    "the vendor states them. Take prices from the vendor's own pricing page; a "
    "figure quoted by a third party may be stale, and if that is all you have, "
    "record it as unconfirmed rather than as the price.",
    "The conditions attached to each price: seat minimums, annual-billing "
    "requirements, usage tiers, whether the figure is introductory. A price "
    "stripped of its conditions is a different, usually cheaper, offer.",
    "Whether a free tier, a trial and an enterprise tier exist. Use null when "
    "the pages do not say — do not infer a free tier from the absence of one.",
    "The pricing model — per seat, usage based, tiered — only if the pages state "
    "it. Otherwise leave the not-public sentinel in place.",
    "Note the exact page the figures came from, so a reader can check them.",
]

PURPOSE = (
    "Extract this company's published pricing from the vendor's own pages, with "
    "the conditions attached to each figure."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"company": "Acme Cloud", "pricing_model": "per seat per month", '
            '"currency": "USD", "has_free_tier": true, '
            '"plans": [{"name": "Team", "price": 10, "unit": "seat/month"}]}'
        ),
    )
