"""Company identification: deciding which company a domain actually is.

Search results are full of entities that merely resemble the target — a
directory entry, a similarly-named firm in another country, a vendor whose
product is called the same thing. Attributing a page to the wrong company
poisons every downstream claim, because every later step trusts the identity
decided here. The prompt therefore asks for the signals that establish identity
and for the disambiguating detail, not for a confident guess.
"""

from __future__ import annotations

from app.prompts.base import PromptContext, build_messages
from app.schemas.company import CompanyIdentity
from app.schemas.provider import ChatMessage

TASK_TYPE = "company_identification"
OUTPUT_SCHEMA = CompanyIdentity

SECTIONS = [
    "The legal or trading name of the company that operates this domain.",
    "The signals that establish which company it is: what it sells, who it "
    "sells to, where it is based, and any distinguishing detail.",
    "Anything that would suggest this is a different company with a similar name "
    "— a parent, a reseller, a directory, an unrelated firm.",
    "If the identity is not established by the evidence, say so rather than "
    "picking the most plausible candidate. An unresolved identity is a valid "
    "outcome and is far cheaper than a wrong one.",
]

PURPOSE = (
    "Identify the company behind the retrieved pages, and state the evidence that "
    "establishes the identification."
)


def build(ctx: PromptContext) -> list[ChatMessage]:
    return build_messages(
        purpose=PURPOSE,
        context=ctx,
        schema=OUTPUT_SCHEMA,
        sections=SECTIONS,
        schema_example=(
            '{"name": "Acme Cloud", "domain": "acme.com", '
            '"description": "Seat-based cloud monitoring", "country": "US"}'
        ),
    )
