"""Retrieval over the run's own documents: chunk, index, filter, rank, rerank.

Layer 3 in the architecture (see `tests/unit/test_layering.py`): it may use
`core`, `schemas` and `utils`, and nothing above it. In particular it does not
know about agents, services or LangGraph, which is what lets an analyst be tested
against a hand-built index with no graph and no provider.
"""

from app.rag.chunking import chunk_document, chunk_documents
from app.rag.indexing import InvertedIndex, build_index, tokenize
from app.rag.metadata import (
    FILTERABLE_KEYS,
    UnknownFilterKeyError,
    chunk_value,
    matches,
    matches_all,
    validate_filters,
)
from app.rag.reranking import RerankWeights, rerank
from app.rag.retrieval import all_sources, candidate_positions, retrieve

__all__ = [
    "FILTERABLE_KEYS",
    "InvertedIndex",
    "RerankWeights",
    "UnknownFilterKeyError",
    "all_sources",
    "build_index",
    "candidate_positions",
    "chunk_document",
    "chunk_documents",
    "chunk_value",
    "matches",
    "matches_all",
    "rerank",
    "retrieve",
    "tokenize",
    "validate_filters",
]
