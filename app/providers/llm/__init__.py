"""LLM backends that leave the process.

One class per vendor, both satisfying
:class:`~app.providers.base.LLMProvider`. They differ in how a model is named and
how a structured answer is requested, and agree on the things that matter to the
rest of the system:

- a response that does not satisfy the requested ``schema`` is an error, never
  prose returned where JSON was asked for;
- usage and duration are recorded on the response, because Task 13's cost
  accounting reads them from there rather than from the SDK;
- a vendor failure is wrapped in :class:`~app.core.exceptions.ProviderError` so
  the circuit breaker in :mod:`app.providers.base` sees one exception type.
"""

from app.providers.llm.gemini import GeminiLLMProvider
from app.providers.llm.openai import OpenAILLMProvider

__all__ = ["GeminiLLMProvider", "OpenAILLMProvider"]
