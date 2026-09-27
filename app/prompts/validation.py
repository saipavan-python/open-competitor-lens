"""Competitor validation: deciding whether a candidate really competes.

This is the narrowest gate in the system and the one most likely to be got
wrong in the permissive direction. A candidate admitted on the strength of a
search snippet produces a comparison row built from a directory listing that
somebody else wrote about a company that may not compete at all — and the row
looks exactly as well-sourced as a row built from the vendor's own pricing page.

So the prompt requires a named signal per candidate, requires that signal to
point at a source in the evidence block, and treats "I know this company" as
worthless. A candidate that cannot clear one primary-source-backed signal is
expected to come back rejected, with the reason recorded.
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.schemas.competitor import ValidatedCompetitor
from app.schemas.provider import ChatMessage

TASK_TYPE = "competitor_validation"
OUTPUT_SCHEMA = ValidatedCompetitor

SECTIONS = [
    "The signals that show this candidate competes: same buyer, same problem, a "
    "product a buyer could switch to, evidence both appear in the same "
    "marketplaces or comparison lists.",
    "For each signal, the source ids from the evidence block that support it. A "
    "signal with no source id is not a signal.",
    "The stated reason this candidate belongs in the comparison, in one sentence.",
    "If the evidence does not establish that it competes, reject it and record "
    "why. Rejection is a normal, expected result and is recorded rather than "
    "hidden — a validation step that never rejects anything is not validating.",
    "Never validate the target company, its parent, or a reseller as its own "
    "competitor.",
]

PURPOSE = (
    "Decide whether this candidate genuinely competes with the target, on named "
    "signals backed by the retrieved sources — or reject it and say why."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"name": "Beta Metrics", "relevance_reason": "Seat-based monitoring '
            'sold to the same platform engineering buyer", '
            '"signals": [{"signal": "listed as an alternative on a review '
            'platform", "supported": true, "source_ids": ["src-7"]}]}'
        ),
    )
