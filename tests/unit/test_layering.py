"""Layering contract, enforced by AST rather than trust.

The plan forbids a module in layer *N* importing layer *N+1*. That rule is what
keeps providers importable without pulling in the orchestrator, and it only
holds if something checks it. These tests walk every module in ``app/`` and
fail on the first upward import, so a violation is caught by CI instead of by a
reviewer noticing it months later.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[2] / "app"

#: Ordered from lowest to highest. A module may import its own layer and any
#: layer below it, never a layer above.
LAYERS: dict[str, int] = {
    "core": 0,
    "schemas": 1,
    "utils": 2,
    "models": 3,
    "repositories": 4,
    "providers": 5,
    "services": 6,
    "agents": 7,
    "graphs": 7,
    "prompts": 7,
    "api": 8,
    "cli": 8,
}


def _layer_of(path: Path) -> str | None:
    relative = path.relative_to(APP_ROOT)
    return relative.parts[0] if relative.parts else None


def _imported_layers(tree: ast.AST, current_layer: str) -> list[tuple[str, int]]:
    """Every ``app.<layer>`` import in a module, with the layer's index."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # level=1 means a relative import inside the same layer, which is
            # allowed; level=0 with a package attr is what we need to check.
            if node.level:
                continue
            names = [node.module or ""]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == "app" and parts[1] in LAYERS:
                found.append((parts[1], LAYERS[parts[1]]))
    del current_layer
    return found


def _python_files() -> list[Path]:
    """Every module in the app, including package ``__init__`` files.

    ``__init__.py`` is included deliberately even though the filter otherwise
    skips underscore-prefixed names: a package's ``__init__`` is where
    re-exports live, which makes it the single most likely place for an upward
    import to hide. Skipping it left the layering guarantee unproven for exactly
    the files that gather a layer's public surface.
    """
    return sorted(
        path
        for path in APP_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
        # `app/__init__.py` is the package root and belongs to no layer; only
        # the layer packages below it are subject to the ordering rule.
        and len(path.relative_to(APP_ROOT).parts) > 1
        and (path.name == "__init__.py" or not path.name.startswith("_"))
    )


def test_the_layer_table_covers_every_package_that_exists() -> None:
    """An unlisted package would silently escape the layering check."""
    present = {path.relative_to(APP_ROOT).parts[0] for path in _python_files()}
    unknown = present - set(LAYERS) - {"__pycache__"}
    assert not unknown, f"packages with no declared layer: {sorted(unknown)}"


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_module_imports_a_higher_layer(path: Path) -> None:
    layer = _layer_of(path)
    assert layer in LAYERS, f"{path} is in undeclared layer {layer!r}"
    own = LAYERS[layer]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for imported, level in _imported_layers(tree, layer):
        assert level <= own, (
            f"{path.relative_to(APP_ROOT)} (layer '{layer}') imports "
            f"'{imported}' (layer {level}); upward imports are forbidden"
        )


def test_providers_do_not_import_services_or_higher() -> None:
    """Providers are the leaf of the outbound side; nothing above may leak in."""
    provider_files = [
        path for path in _python_files() if _layer_of(path) == "providers"
    ]
    assert provider_files, "expected provider modules to exist"
    for path in provider_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for imported, level in _imported_layers(tree, "providers"):
            assert level <= LAYERS["providers"], (
                f"{path.relative_to(APP_ROOT)} imports '{imported}'"
            )


def test_utils_does_not_import_providers_or_services() -> None:
    """Utilities are shared by everything, so they must stay dependency-free."""
    for path in _python_files():
        if _layer_of(path) != "utils":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for imported, level in _imported_layers(tree, "utils"):
            assert level <= LAYERS["utils"], (
                f"{path.relative_to(APP_ROOT)} imports '{imported}'"
            )
