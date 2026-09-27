"""Product analyst: the offering, and how it is positioned.

The distinction this analyst has to hold is between what a company *lists* and
what it *does*. A product page is the vendor's own claim about its own product,
which is evidence of the claim and only evidence of the claim. Treating a
marketing page's description of a feature as proof the feature works is how a
comparison report ends up saying two products are equivalent because both
vendors describe them as "enterprise-grade".

So the queries target the pages that enumerate things — features, integrations,
use cases — and the prompt is the one that separates observation from the
vendor's framing of it.
"""

from __future__ import annotations

from app.agents.analysis_base import CategoryAnalyst
from app.schemas.products import ProductAnalysis


class ProductAnalyst(CategoryAnalyst):
    """Products, features, segments and positioning."""

    name = "product_analysis"
    category = "product"
    prompt_name = "product_analysis"
    output_schema = ProductAnalysis
    goal = "Catalogue what the company offers and how it is positioned."
    retrieval_queries = (
        "products features",
        "solutions use cases",
        "integrations platform compatibility",
        "differentiators compared to alternatives",
    )
