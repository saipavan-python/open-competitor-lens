"""SEO analysis: technical signals read off the retrieved markup.

Almost every field here is a mechanical fact about the HTML — a canonical tag, a
robots directive, a count of sitemap URLs — and those are the wrong kind of
question to put to a language model at all. A model asked "does this site have
structured data" will answer from the shape of a typical commercial site, which
is precisely the kind of plausible-and-unsourced answer the rest of the system
exists to prevent.

So this prompt is written for a reader that may be filling fields from
convention rather than from the retrieved markup. Each field is to be filled
only from what is in the evidence, left null when the page was not retrieved,
and the unknowns list is expected to be long. A null here is cheap; a fabricated
canonical tag is a defect that no later step can detect.
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.schemas.provider import ChatMessage
from app.schemas.seo import SeoAnalysis

TASK_TYPE = "seo_analysis"
OUTPUT_SCHEMA = SeoAnalysis

SECTIONS = [
    "Only the technical signals visible in the retrieved pages. Do not infer a "
    "signal from what a site of this type would normally have — a canonical tag "
    "you expected but did not see is an absent field, not a present one.",
    "The pages actually checked, listed by URL, so the coverage of this analysis "
    "is visible. An SEO analysis based on one page is a one-page analysis, and "
    "the report should say so.",
    "Title tags, meta descriptions and heading structure as they appear.",
    "Robots directives, sitemap and structured data, only where the retrieved "
    "content shows them.",
    "Everything not established, in the unknowns list. Expect this list to be "
    "long: it is the honest description of what a handful of pages can and "
    "cannot tell you about a site's search strategy.",
]

PURPOSE = (
    "Report the technical SEO signals present in the retrieved pages, and nothing "
    "about a site you did not retrieve."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"company": "Acme Cloud", "title_tag": "Cloud monitoring", '
            '"has_canonical": true, "pages_checked": ["https://acme.com/"], '
            '"unknowns": ["sitemap contents", "robots directives"], '
            '"confidence": "low"}'
        ),
    )
