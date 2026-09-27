"""Market analyst: the market both companies are in.

The one category that is inherently comparative, which changes what it can be
grounded in. A market trend is not a fact any single vendor's website states — it
is a pattern across several of them, or an inference from their positioning. So
this analyst's evidence is necessarily weaker than a pricing figure's, and its
schema is correspondingly explicit: `insights` for what follows from the
evidence, `trends` for what several sources agree on, and `unknowns` for the
questions no page in the corpus answers.

Its queries are also phrased against the category rather than the company. Every
other analyst retrieves one company's own pages; this one retrieves the market,
so asking it for "our products" would return the same vendor pages the product
analyst already read and produce a confident description of one company
presented as industry analysis.
"""

from __future__ import annotations

from app.agents.analysis_base import CategoryAnalyst
from app.schemas.market import MarketAnalysis


class MarketAnalyst(CategoryAnalyst):
    """Market size, trends, and the gaps in it."""

    name = "market_analysis"
    category = "market"
    prompt_name = "market_analysis"
    output_schema = MarketAnalysis
    goal = "Characterise the market, its trends and its gaps."
    retrieval_queries = (
        "market size trends growth",
        "industry analysis report landscape",
        "customer needs pain points unmet",
        "new entrants emerging alternatives",
    )
