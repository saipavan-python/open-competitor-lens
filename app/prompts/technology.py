"""Technology analysis: stack signals, which are mostly indirect.

A technology stack is rarely published. It is inferred from a careers page, a
bundled JavaScript filename, a disclosed dependency, a status-page integration.
That makes this category structurally inference-heavy, and it is why the schema
separates ``observations`` from ``inferences`` and why every signal is expected
to name the page it came from.

The prompt is explicit that a careers posting is evidence a company uses a
technology, not a specification of how they run it in production — a distinction
that matters when a report recommends matching a competitor's architecture.
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.schemas.provider import ChatMessage
from app.schemas.technology import TechnologyAnalysis

TASK_TYPE = "technology_analysis"
OUTPUT_SCHEMA = TechnologyAnalysis

SECTIONS = [
    "The technology signals present in the retrieved pages, each naming the page "
    "it came from: disclosed dependencies, integrations, status-page vendors, "
    "assets referenced by the site, engineering job postings.",
    "The categories the signals cover, and — equally important — the categories "
    "they do not. A careers page reveals the interesting stack and says nothing "
    "at all about deployment, and a reader who does not know that will read the "
    "gap as absence of signal rather than absence of data.",
    "Observations: what the pages state. Inferences: what you conclude from those "
    "statements. A job posting mentioning a database is an observation; the "
    "company runs on that database is an inference, and belongs in the second "
    "list.",
    "The number of job postings seen and the GitHub organisations found, which "
    "are counts of what was retrieved and not estimates of headcount.",
    "What the retrieved pages cannot tell you, in the unknowns list.",
]

PURPOSE = (
    "Identify technology signals from the retrieved pages, separating what the "
    "pages state from what you infer from it."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"company": "Acme Cloud", "signals": [{"technology": "Postgres", '
            '"category": "database", "source_ids": ["src-4"]}], '
            '"observations": ["careers page lists Postgres"], '
            '"inferences": ["primary datastore is Postgres"], '
            '"unknowns": ["deployment platform"], "confidence": "low"}'
        ),
    )
