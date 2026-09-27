"""Outbound provider adapters and the protocols they satisfy.

Concrete vendor implementations live in subpackages (``web``, ``llm``, …). This
package-level module holds only the protocols, the circuit breaker and the
deterministic offline substitutes, so importing a protocol never drags in a
vendor SDK.
"""

from app.providers.base import (
    BrowserProvider,
    EmbeddingProvider,
    GraphStore,
    LLMProvider,
    ProviderHealth,
    SearchProvider,
    VectorStore,
    WebProvider,
)
from app.providers.registry import ProviderKind, ProviderRegistry, ProviderSpec, get_registry

__all__ = [
    "BrowserProvider",
    "EmbeddingProvider",
    "GraphStore",
    "LLMProvider",
    "ProviderHealth",
    "ProviderKind",
    "ProviderRegistry",
    "ProviderSpec",
    "SearchProvider",
    "VectorStore",
    "WebProvider",
    "get_registry",
]
