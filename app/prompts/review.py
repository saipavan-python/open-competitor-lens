"""Review analysis: what customers say, attributed and counted.

Two things distinguish this from every other analyst. Reviews are second-hand
accounts, so a review is evidence of *a customer's experience* and not of the
product's behaviour — the distinction the observation/inference split protects
again. And a theme asserted without its underlying reviews is the most
attractive thing a model can produce here: "users complain about onboarding" is
a sentence it can write from having read three reviews, and it is exactly the
kind of claim a reader will act on.

So themes must carry the reviews they came from, a single review may not become
a pattern, and the repeated-complaint fields require more than one mention.
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.schemas.provider import ChatMessage
from app.schemas.reviews import ReviewAnalysis

TASK_TYPE = "review_analysis"
OUTPUT_SCHEMA = ReviewAnalysis

SECTIONS = [
    "The platforms covered, with the number of reviews behind each. A theme drawn "
    "from two reviews is an anecdote and must be reported as one.",
    "The themes present, each with the reviews it was drawn from, split into "
    "positive and negative. A theme with no supporting review is not a theme.",
    "Repeated complaints, feature requests, support issues and pricing "
    "complaints — only where the same point appears more than once. If a single "
    "review raises something, it belongs in observations, not in the repeated "
    "list.",
    "Observations about what reviewers say, kept separate from any inference "
    "about the product. A reviewer's account is evidence of their experience, "
    "not a verified fact about the software.",
    "Platforms and themes the retrieved reviews do not cover, in the unknowns "
    "list. Review coverage is heavily biased toward a few large platforms, and "
    "the bias is more interesting than any individual rating.",
]

PURPOSE = (
    "Analyse customer reviews from the retrieved quotes, attributing every theme "
    "to the reviews behind it."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"company": "Acme Cloud", "platforms": [{"platform": "g2", '
            '"review_count": 40}], "repeated_complaints": ["slow dashboard load"], '
            '"unknowns": ["reviews from smaller platforms"], "confidence": "medium"}'
        ),
    )
