"""The prompt registry.

Every prompt in the system is registered here, which makes the set of prompts a
closed, enumerable thing rather than something discovered by grepping for
``build``. That matters for two reasons: the prompt tests iterate this registry,
so a prompt added without a matching contract is immediately visible, and the
orchestrator can look up a prompt by name without importing seventeen modules
and guessing which one it wanted.

The registry is a mapping of name to module, and nothing in it holds a model id
or a credential — routing is :class:`~app.services.model_router.ModelRouter`'s
job, and keeping the two separate is what lets a prompt be unit-tested without a
model and a model be swapped without touching a prompt.
"""

from __future__ import annotations

from types import ModuleType

from app.prompts import (
    company,
    comparison,
    discovery,
    final_report,
    market,
    marketing,
    planning,
    pricing,
    product,
    query,
    review,
    seo,
    social,
    synthesis,
    technology,
    validation,
    website,
)
from app.prompts.base import PromptContext
from app.schemas.provider import ChatMessage

#: Every prompt module, by the name the orchestrator refers to it by.
PROMPTS: dict[str, ModuleType] = {
    module.TASK_TYPE: module
    for module in (
        planning,
        company,
        discovery,
        validation,
        website,
        product,
        pricing,
        marketing,
        seo,
        social,
        review,
        technology,
        market,
        query,
        comparison,
        synthesis,
        final_report,
    )
}


def names() -> list[str]:
    """The registered prompt names, in a stable order."""
    return sorted(PROMPTS)


def get(name: str) -> ModuleType:
    """The prompt module registered under ``name``."""
    try:
        return PROMPTS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown prompt {name!r}. Registered: {', '.join(names())}."
        ) from exc


def build(name: str, ctx: PromptContext) -> list[ChatMessage]:
    """Render a registered prompt against a context."""
    return get(name).build(ctx)


__all__ = [
    "PROMPTS",
    "PromptContext",
    "build",
    "get",
    "names",
]
