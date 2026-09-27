"""SEO analyst: signals that can be observed rather than guessed.

Every field in this analysis is either read off a page or is not recorded. That
is the whole discipline of the category: a rank, a keyword difficulty score or a
"domain authority" figure is a number a third party computed by a method nobody
in the report can check, and presenting one as a finding is exactly the
fabrication this system is built to prevent. There is a schema for SEO that is
mostly such numbers, and this analyst does not fill it in.

What is left is genuinely useful and genuinely checkable: whether a canonical tag
exists, whether a sitemap and robots.txt were reachable, which structured-data
types a page declares, whether the vendor maintains a blog at all. A reader can
open the same URLs and see the same thing.

The fields that could not be observed stay ``None``, and the analyst's
`unknowns` say why — usually that the page was never retrieved, which is a
different fact from "the company has no sitemap".
"""

from __future__ import annotations

from app.agents.analysis_base import CategoryAnalyst
from app.schemas.seo import SeoAnalysis


class SeoAnalyst(CategoryAnalyst):
    """Observable on-page and technical signals."""

    name = "seo_analysis"
    category = "seo"
    prompt_name = "seo_analysis"
    output_schema = SeoAnalysis
    goal = "Record observable SEO and technical signals."
    retrieval_queries = (
        "sitemap robots canonical",
        "meta description title tag",
        "structured data schema markup",
        "blog resources guides articles",
    )
