"""No dead code: every definition is used by the code it belongs to.

* **Python**: every function, class, method and module-level constant of ``src/`` and ``tools/``
  is referenced by the code of ``src/`` or ``tools/``; one only the tests use belongs in
  ``tests/helpers.py``. A helper of ``tests/`` is referenced by some test. A reference is a name,
  an attribute, an import, or a word of a string (annotations in quotes, names looked up by
  ``getattr``). Special methods, decorated functions (routes, fixtures, validators, properties),
  tests and :data:`FRAMEWORK_HOOKS` are called by their framework, and exempt.
* **JavaScript**: every ``export`` is imported by another module of the page or by a Node test;
  a name only its own module uses is not exported.
"""

import ast
import re
from collections import Counter
from typing import TYPE_CHECKING

from test_conventions import ROOT, SCRIPT_SUFFIXES, project_files, python_files

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

#: Functions a framework calls by name: ``torch.nn.Module.__call__`` runs ``forward``,
#: and ``ast.NodeTransformer`` runs ``visit_<node type>``.
FRAMEWORK_HOOKS = {"forward", "visit_Name"}
#: Top-level directories whose definitions must be used outside the tests.
PRODUCT_DIRS = {"src", "tools"}
TEST_DIR = "tests"
_TEST_PREFIX = "test_"
_DUNDER = "__"
_WORD = re.compile(r"[A-Za-z_]\w*")
_EXPORT = re.compile(r"^export (?:async )?(?:function\*?|const|let|class) (\w+)", re.MULTILINE)


def references(sources: "Iterable[str]") -> Counter[str]:
    """Count the names the given Python sources use.

    Args:
        sources: Python source texts.

    Returns:
        How often each name is read, as a name, an attribute, an import or a word of a string.
    """
    seen: Counter[str] = Counter()
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                seen[node.id] += 1
            elif isinstance(node, ast.Attribute):
                seen[node.attr] += 1
            elif isinstance(node, ast.alias):
                seen[node.name.rsplit(".", 1)[-1]] += 1
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                seen.update(_WORD.findall(node.value))
    return seen


def definitions(source: str) -> list[tuple[str, int]]:
    """List what a Python source defines and must have a user.

    Args:
        source: Python source text.

    Returns:
        ``(name, line)`` of its functions, classes, methods and module-level constants, the
        exempt ones (see the module docstring) left out.
    """
    tree = ast.parse(source)
    found = [
        (node.name, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not _is_exempt(node)
    ]
    found.extend(_constants(tree))
    return found


def _is_exempt(node: "ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef") -> bool:
    """Tell whether a definition needs no caller in the product (see the module docstring).

    Args:
        node: The definition.

    Returns:
        Whether it is decorated, special, a test or a framework hook.
    """
    return bool(
        node.decorator_list
        or node.name.startswith((_DUNDER, _TEST_PREFIX))
        or node.name in FRAMEWORK_HOOKS
    )


def _constants(tree: ast.Module) -> list[tuple[str, int]]:
    """List a module's top-level assigned names (special ones left out).

    Args:
        tree: The parsed module.

    Returns:
        ``(name, line)`` pairs.
    """
    found: list[tuple[str, int]] = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        found.extend(
            (t.id, node.lineno)
            for t in targets
            if isinstance(t, ast.Name) and not t.id.startswith(_DUNDER)
        )
    return found


def unreferenced(defining: "dict[str, str]", using: "Iterable[str]") -> list[str]:
    """Find the definitions that no user source references.

    Args:
        defining: Python source text by its location (for the report).
        using: The sources whose references count.

    Returns:
        ``location:line name`` of every definition nothing references.
    """
    seen = references(using)
    return [
        f"{where}:{line} {name}"
        for where, source in defining.items()
        for name, line in definitions(source)
        if not seen[name]
    ]


def unimported_exports(scripts: "dict[str, str]") -> list[str]:
    """Find the JavaScript exports that no other module imports.

    Args:
        scripts: Script text by its location.

    Returns:
        ``location name`` of every export no other script names.
    """
    return [
        f"{where} {name}"
        for where, text in scripts.items()
        for name in _EXPORT.findall(text)
        if not any(
            re.search(rf"\b{name}\b", other) for there, other in scripts.items() if there != where
        )
    ]


def _texts(paths: "Iterable[Path]") -> dict[str, str]:
    """Read source files, keyed by their path from the repository root.

    Args:
        paths: Files to read.

    Returns:
        Text by relative path.
    """
    return {str(p.relative_to(ROOT)): p.read_text(encoding="utf-8") for p in paths}


def test_every_product_definition_is_used_by_the_product() -> None:
    """Nothing in src/ or tools/ is dead, or kept alive by the tests alone."""
    product = _texts(p for p in python_files() if p.relative_to(ROOT).parts[0] in PRODUCT_DIRS)
    assert unreferenced(product, product.values()) == []


def test_every_test_helper_is_used() -> None:
    """Every helper of tests/ serves some test."""
    everything = _texts(python_files())
    helpers = {where: text for where, text in everything.items() if where.startswith(TEST_DIR)}
    assert unreferenced(helpers, everything.values()) == []


def test_every_script_export_is_imported() -> None:
    """Every JavaScript export is imported by another module or a Node test."""
    scripts = _texts(p for p in project_files() if p.suffix in SCRIPT_SUFFIXES)
    assert unimported_exports(scripts) == []


def test_planted_dead_code_is_found() -> None:
    """The rules catch what they claim to, and spare what is exempt."""
    product = (
        "LIMIT = 3\nUNUSED = 4\n"
        + "def used():\n    return LIMIT\n"
        + "def dead():\n    return used()\n"
        + "def hinted() -> 'Kept': ...\n"
        + "class Kept:\n    def __len__(self): return 0\n"
        + "".join(f"    def {hook}(self): ...\n" for hook in sorted(FRAMEWORK_HOOKS))
    )
    assert unreferenced({"m.py": product}, [product]) == [
        "m.py:5 dead",
        "m.py:7 hinted",
        "m.py:2 UNUSED",
    ]
    scripts = {"a.js": "export function shown() {}\nexport const alone = 1;", "b.js": "shown();"}
    assert unimported_exports(scripts) == ["a.js alone"]
