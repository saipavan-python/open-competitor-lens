"""Pricing analyst: the numbers a reader is most likely to act on.

Two failure modes matter more here than anywhere else.

**A third-party price is not the current price.** Prices quoted by a review site
or a cached comparison table go stale silently, and a report that presents one as
the price is wrong in the way that costs a reader money. The vendor's own page is
the only acceptable source, and a third-party figure that is all there is should
come back marked as unconfirmed — which is a different thing from "the price is
$20", and is the distinction this analyst exists to preserve.

**"Starts at" is not a price.** A price list with annual and monthly rates, a
seat minimum and a usage tier contains several prices. Reporting the lowest as
*the* price misstates the offer, sometimes by a factor of ten.

The retrieval queries therefore carry vendors' own vocabulary — "plans",
"packages", "per seat" — rather than the category's name. A vendor that calls its
price list "Plans" never uses the word "pricing" on the page a crawler reaches,
and an analyst that searched for "pricing" would report the single most
consequential gap in the report as genuinely absent.
"""

from __future__ import annotations

from app.agents.analysis_base import CategoryAnalyst
from app.schemas.pricing import PricingAnalysis


class PricingAnalyst(CategoryAnalyst):
    """Published pricing, with the conditions attached to each figure."""

    name = "pricing_analysis"
    category = "pricing"
    prompt_name = "pricing_analysis"
    output_schema = PricingAnalysis
    goal = "Record published pricing with the conditions attached to it."
    retrieval_queries = (
        "pricing plans packages",
        "cost per seat per month",
        "free tier trial enterprise plan",
        "billing period annual monthly",
    )
