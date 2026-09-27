"""Synthesis: turning findings into insights without inflating them.

Every analyst produces observations. Synthesis is the step that decides which of
them mean something, and it is the step most likely to overreach, because an
insight is by definition a step beyond the evidence and there is always a more
interesting one available.

The ``Insight`` schema helps: ``observation`` and ``interpretation`` are separate
fields, and the prompt is written so the observation can be read on its own. An
insight whose observation does not stand up on its own is a conclusion looking
for a fact, and the report should not carry it.
"""

from __future__ import annotations

from pydantic import Field

from app.prompts.base import PromptContext, build_messages
from app.schemas.common import DomainModel
from app.schemas.finding import Insight
from app.schemas.provider import ChatMessage

TASK_TYPE = "synthesis"

SECTIONS = [
    "The observations worth carrying forward, drawn from the evidence in front of "
    "you and from nothing else.",
    "For each, the interpretation: what the observation means for a buyer "
    "comparing these companies. The interpretation is your reasoning and is "
    "offered as such.",
    "The evidence backing each insight, by source id. An insight with no evidence "
    "is a guess, and a guess phrased as an insight is the most damaging thing "
    "this system can emit — it is the one output a reader is unlikely to check.",
    "An honest confidence. Most cross-company insights from partial evidence are "
    "low confidence, and a synthesis full of high-confidence insights is a sign "
    "the confidence was chosen rather than earned.",
    "The findings you considered and rejected as unsupported, rather than "
    "quietly omitting them. A rejected finding is information about the limits of "
    "the analysis.",
]


class SynthesisBundle(DomainModel):
    """The synthesis step's output."""

    insights: list[Insight] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)


OUTPUT_SCHEMA = SynthesisBundle

PURPOSE = (
    "Draw the insights the evidence supports, with each one's observation, "
    "interpretation and backing sources kept distinct."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"insights": [{"type": "advantage", '
            '"observation": "Acme publishes per-seat pricing; neither competitor '
            'publishes any", "interpretation": "price is usable as a lever at '
            'renewal", "evidence": ["src-1"], "confidence": "medium"}], '
            '"rejected": ["Acme has the cheapest price - no competitor figures '
            'were published"]}'
        ),
    )
