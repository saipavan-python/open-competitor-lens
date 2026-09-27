"""Comparison: putting competitors side by side without filling the gaps.

A comparison table is where partial data does the most damage, because a table
implies that every cell was checked. When one vendor publishes prices and three
do not, the interesting table is the one that says so — and a model asked to
"compare pricing across competitors" will produce four populated rows, sourcing
the three unpublished figures from whichever comparison article it half-remembers.

Hence the unknown sentinel is the load-bearing instruction in this prompt, and
the evidence-quality scores are asked for honestly: a dimension where one
company is well covered and another is not is a dimension where the comparison
is weak, and the report needs to be able to say that.
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.schemas.comparison import ComparisonMatrix
from app.schemas.provider import ChatMessage

TASK_TYPE = "comparison"
OUTPUT_SCHEMA = ComparisonMatrix

SECTIONS = [
    "The dimensions worth comparing, chosen because the evidence supports a "
    "comparison on them — not because they are standard in this industry.",
    "For each dimension, each competitor's value. A company that has not "
    "published a value gets the unknown sentinel. Never fill a competitor's cell "
    "from a third party's comparison table, from a similar company, or from what "
    "vendors in this category usually do — a filled cell is indistinguishable "
    "downstream from a sourced one.",
    "The evidence quality per company: how well the retrieved evidence actually "
    "covers that company. A dimension with one well-covered company and three "
    "thin ones is not a comparison, and the scores are how the report says so.",
    "Notes on any dimension where the competitors' figures are not actually "
    "like-for-like — different billing periods, different seat minimums, "
    "different units. Apparent price differences are frequently an artefact of "
    "comparing two different things.",
]

PURPOSE = (
    "Build a side-by-side comparison of the competitors on the dimensions the "
    "evidence supports, marking everything else unknown."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"companies": ["Acme Cloud", "Beta Metrics"], '
            '"dimensions": [{"field": "entry_price", "label": "Entry price", '
            '"category": "pricing", '
            '"values": {"Acme Cloud": "$10/seat/mo", "Beta Metrics": "unknown"}}], '
            '"evidence_quality": {"Acme Cloud": 0.9, "Beta Metrics": 0.2}, '
            '"notes": ["Beta Metrics does not publish prices"]}'
        ),
    )
