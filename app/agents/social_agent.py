"""Social analyst: which presences exist, and what they show.

A presence that could not be retrieved is recorded as not found, not as absent.
The distinction is not pedantry: several companies simply do not link their
social accounts from any page a crawler reaches, and a report that concludes
"no social presence" from that is confidently wrong. `not_found` and an empty
`profiles` list mean different things and the schema keeps them apart.

Follower counts are included but treated as the volatile, third-party-numbered
they are — they change without the page changing, and a count quoted without its
date is a fact with an expiry date and no expiry printed on it.
"""

from __future__ import annotations

from app.agents.analysis_base import CategoryAnalyst
from app.schemas.social import SocialAnalysis


class SocialAnalyst(CategoryAnalyst):
    """Social presences and the signals visible on them."""

    name = "social_analysis"
    category = "social"
    prompt_name = "social_analysis"
    output_schema = SocialAnalysis
    goal = "Identify the company's social presence and engagement signals."
    retrieval_queries = (
        "twitter x linkedin facebook instagram youtube",
        "follow us social media links",
        "community forum discord",
    )
