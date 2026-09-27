"""Search backends.

Every search result is a lead. None of them is a retrieved source, and the
evidence gate refuses to cite one.
"""

from app.providers.search.tavily import TavilySearchProvider

__all__ = ["TavilySearchProvider"]
