"""Review analyst: what customers say, kept apart from what vendors claim.

The only category in the report built almost entirely from third-party text, and
the only one where the source's own bias is the subject rather than a caveat. A
review platform's users are self-selected: the angriest and the happiest are
over-represented, and a summary that reports "sentiment is mixed" from those
pages is reporting the platform's selection effect as a finding about the
product.

Two consequences shape the retrieval. Queries name the review platforms
explicitly, because review text on a vendor's own testimonials page is
marketing wearing a review's clothes — the most common way a report ends up
citing a vendor's own homepage as a customer's independent opinion. And the
thematics are separated into positive and negative rather than netted into a
score, because a single sentiment number hides exactly the detail a reader
comparing two products needs.
"""

from __future__ import annotations

from app.agents.analysis_base import CategoryAnalyst
from app.schemas.reviews import ReviewAnalysis


class ReviewAnalyst(CategoryAnalyst):
    """Customer sentiment from third-party review platforms."""

    name = "review_analysis"
    category = "review"
    prompt_name = "review_analysis"
    output_schema = ReviewAnalysis
    goal = "Summarise customer sentiment from third-party review platforms."
    retrieval_queries = (
        "reviews ratings customer feedback",
        "g2 captrona trustradius alternatives",
        "complaints issues problems",
        "feature requests missing features",
    )
