"""Technology analyst: the stack, evidenced rather than inferred.

Stack detection is the most inviting place in a competitor report to invent
something. The badge on a vendor's homepage is a claim; a JavaScript bundle names
the frameworks the marketing site is built with, which is usually not the stack
that runs the product; a job posting naming a tool is the strongest signal
available, because a company hires for the stack it runs.

So the queries deliberately prefer artefacts where a technology named in the
text is a technology in use — job postings, API documentation, changelogs — over
the marketing pages where a technology named in the text is a technology
mentioned. Where a signal is a mention rather than a use, it is recorded as a
signal with its detail, not promoted to a fact about the product's internals.
"""

from __future__ import annotations

from app.agents.analysis_base import CategoryAnalyst
from app.schemas.technology import TechnologyAnalysis


class TechnologyAnalyst(CategoryAnalyst):
    """Technologies the company is evidenced to use."""

    name = "technology_analysis"
    category = "technology"
    prompt_name = "technology_analysis"
    output_schema = TechnologyAnalysis
    goal = "Identify the technologies the company is evidenced to use."
    retrieval_queries = (
        "engineering stack technologies",
        "job postings careers engineering hiring",
        "api documentation sdk developers",
        "changelog release notes github",
    )
