"""Market analysis: the category, and what the evidence does not settle.

Market analysis is the most tempting place in the whole system to write from
general knowledge, because the model has read a great deal about markets and
none of it is in the evidence block. "This is a growing market with high
competition" is a sentence that requires no source and will be produced whether
or not anything was retrieved.

The prompt therefore asks for trends and insights tied to named sources, and
treats the opportunity and risk fields as the model's own reasoning rather than
findings. It also asks for the segments the evidence does *not* cover, because in
a market analysis the shape of the gap is usually the actionable part.
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.schemas.market import MarketAnalysis
from app.schemas.provider import ChatMessage

TASK_TYPE = "market_analysis"
OUTPUT_SCHEMA = MarketAnalysis

SECTIONS = [
    "The market as the retrieved sources define it: its boundaries, geography and "
    "industry. If the sources disagree about what counts as the market, report "
    "the disagreement rather than picking one.",
    "Trends, each tied to the sources that evidence it. A trend you believe is "
    "true of this industry but cannot source is not a trend in this report.",
    "Customer needs, emerging features, common business models and pricing "
    "patterns, as the sources state them.",
    "Underserved segments and new entrants named in the evidence — and the "
    "segments the evidence says nothing about, which is usually the more useful "
    "of the two.",
    "The opportunity and the reasoning behind it, as your inference from the "
    "above, labelled as such and never presented as a sourced finding.",
]

PURPOSE = (
    "Analyse the market from the retrieved sources, separating sourced trends from "
    "your own inference about opportunity."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"market": "cloud monitoring", "industry": "observability", '
            '"trends": [{"trend": "consolidation of point tools", '
            '"source_ids": ["src-9"]}], '
            '"underserved_segments": ["mid-market teams wanting audit logs"], '
            '"unknowns": ["non-English market coverage"], "confidence": "low"}'
        ),
    )
