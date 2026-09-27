"""Marketing analyst: how the company talks about itself.

The useful signal here is not what a company claims but *how* — repeated themes
across pages, the vocabulary it chooses, the customers it puts at the front of
its case studies, the calls to action it repeats. A company whose case studies
are all about a single vertical is telling you something a features list cannot.

The hard part is that marketing pages are entirely the vendor's own framing, so
almost everything retrieved is a claim *about* the vendor rather than a fact
*about* the world. This analyst's schema keeps those apart: `observations` for
what the pages say, `inferences` for what follows, `unknowns` for the rest, and
the distinction matters because a reader acting on an inference should know they
are acting on one.
"""

from __future__ import annotations

from app.agents.analysis_base import CategoryAnalyst
from app.schemas.marketing import MarketingAnalysis


class MarketingAnalyst(CategoryAnalyst):
    """Messaging, channels and demand generation."""

    name = "marketing_analysis"
    category = "marketing"
    prompt_name = "marketing_analysis"
    output_schema = MarketingAnalysis
    goal = "Describe the company's messaging, channels and demand generation."
    retrieval_queries = (
        "about mission vision values",
        "case studies customers results",
        "blog resources guides",
        "contact demo request lead form",
    )
