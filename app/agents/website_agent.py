"""Website analyst: who this company is.

The baseline category. Every other analyst assumes it knows what the company
does, and where it does not, the mistake is invisible — a pricing analysis of a
company that has since pivoted will faithfully report prices for a product it no
longer sells, with full citations.

So this analyst is the one that has to distinguish a company from a division of
it, and a current description from an archived one. Its queries lean towards the
pages a vendor keeps updated ("about", "what we do") rather than the pages it
writes once and forgets (press releases), because a stale about page is the most
common way a report profiles a company that has changed.
"""

from __future__ import annotations

from app.agents.analysis_base import CategoryAnalyst
from app.schemas.company import CompanyProfile


class WebsiteAnalyst(CategoryAnalyst):
    """Who the company is, what it does, and who it is for."""

    name = "website_analysis"
    category = "website"
    prompt_name = "website_analysis"
    output_schema = CompanyProfile
    goal = "Establish what the company is and who it serves."
    retrieval_queries = (
        "about us company overview",
        "what we do products services",
        "who we serve target customers",
        "value proposition why choose us",
    )
