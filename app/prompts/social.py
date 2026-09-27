"""Social analysis: presence on platforms, and honest absence.

The schema has a ``not_found`` field, and it is the important one. Social
presence is the category most likely to produce fabricated output: every company
has a plausible handle on every platform, and a model asked to list a company's
social profiles will produce them. A profile that does not exist, or that belongs
to a different company with the same name, becomes a comparison row that looks
verified because it has a URL.

So absence is a first-class result here. The prompt asks for the platforms
searched and the ones where nothing was found, so that "no presence" is recorded
as a finding rather than filled in.
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.schemas.provider import ChatMessage
from app.schemas.social import SocialAnalysis

TASK_TYPE = "social_analysis"
OUTPUT_SCHEMA = SocialAnalysis

SECTIONS = [
    "The profiles actually found in the evidence, with their platform, handle and "
    "URL. Do not construct a handle from a company name — an unverified profile "
    "URL is worse than no profile, because it looks like a source.",
    "For each profile, evidence that the account belongs to this company and not "
    "to another with the same name.",
    "The platforms that were searched and where nothing was found, in the "
    "not_found list. Recording the absence is the point of this step, not a "
    "failure of it.",
    "Anything the retrieved pages do not establish about audience size, "
    "engagement or posting frequency, in the unknowns list. Follower counts seen "
    "in a snippet are a search result, not a measurement.",
]

PURPOSE = (
    "Report this company's social presence as found in the evidence, and record "
    "the platforms where nothing was found."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"company": "Acme Cloud", "not_found": ["tiktok", "youtube"], '
            '"observations": ["github org linked from the docs site"], '
            '"unknowns": ["follower counts"], "confidence": "low"}'
        ),
    )
