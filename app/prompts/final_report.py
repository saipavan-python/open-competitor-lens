"""The final report: the one output a person will read.

Everything upstream exists to make this step defensible, and this is the prompt
where the cost of a small mistake is highest: a report is quoted, forwarded and
acted on, and a fabricated figure in one is very hard to trace back to the model
that produced it.

Three instructions carry most of the weight. Write from the evidence block, and
when it does not support a point, do not make the point. Report conflicts and
gaps rather than resolving them — a report that quietly picks a winner among
conflicting sources is more misleading than one that admits the disagreement.
And mark inference as inference, because a reader who cannot tell which sentences
are sourced will treat them all as sourced.
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.prompts.common import REPORTING_INSTRUCTION
from app.schemas.provider import ChatMessage
from app.schemas.report import ReportBundle

TASK_TYPE = "final_report"
OUTPUT_SCHEMA = ReportBundle

SECTIONS = [
    "A summary a reader could act on: who these companies are, what the "
    "evidence establishes, and what it does not.",
    "The comparison, with unknown cells shown as unknown. Do not present a "
    "comparison as complete when part of it is unknown — an incomplete table "
    "labelled as complete is a false statement about the analysis.",
    "The findings, each marked as observation or inference, with the sources "
    "behind it.",
    "Where sources disagree, both positions and the disagreement between them. "
    "Resolving a conflict silently is a fabrication of consensus, and it is the "
    "one error here that is invisible in the finished document.",
    "The limits of the analysis: what was not retrieved, what could not be "
    "verified, and which questions remain open. A report that does not state its "
    "own limits invites the reader to supply them as conclusions.",
]

PURPOSE = (
    "Write the final competitor analysis report from the retrieved evidence, "
    "showing conflicts and gaps rather than resolving them."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"target_company": {"name": "Acme Cloud"}, '
            '"product_comparison": {"companies": ["Acme Cloud", "Beta Metrics"]}, '
            '"notes": ["Beta Metrics does not publish pricing"]}'
        ),
        extra_system=REPORTING_INSTRUCTION,
    )
