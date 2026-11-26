"""No duplicated code and no needless wrappers, in Python and JavaScript.

What a review used to catch by eye, checked on every run:

* **Pure forwarders**: a function whose whole body is ``return other(its own parameters, in
  order)`` adds a name and nothing else; callers should call ``other``. Decorated functions
  (caching, properties, validators...), special methods and tests are exempt: they add behaviour.
  So are the methods of :data:`PROTOCOL_ADAPTERS`, which give a library's call the name an
  interface requires.
* **Identical functions**: two functions with the same body, variable names aside.
* **Repeated logic**: the same :data:`RUN_STATEMENTS` statements in a row in two places, variable
  names aside, is logic to share. Runs of plain calls (``x = f(a)``, ``g(b)``, arguments that
  are names or constants) are wiring, not logic, and are allowed: entry points start alike.
* **Repeated script lines**: the same :data:`RUN_LINES` meaningful lines in a row in two places
  of the project's JavaScript.

Size thresholds keep one-liners and signatures out: only blocks with some substance count.
"""

import ast
import re
from collections import defaultdict
from typing import TYPE_CHECKING

from test_conventions import ROOT, SCRIPT_SUFFIXES, SOURCE_DIRS, project_files, python_files

if TYPE_CHECKING:
    from pathlib import Path

#: Methods implementing an interface by delegating to a library (``Class.method``).
PROTOCOL_ADAPTERS = {"OpenCVEngine.upscale"}  # SREngine.upscale over dnn_superres' upsample
#: Statements in a row that make a repeated run of logic.
RUN_STATEMENTS = 3
#: Size (AST dump characters) from which a run, or a function body, has substance.
MIN_RUN_SIZE = 700
MIN_BODY_SIZE = 600
#: Meaningful script lines in a row that make a repeated run, and their smallest total length.
RUN_LINES = 4
MIN_LINES_SIZE = 80
#: Script lines that carry no logic: comments, imports, lone brackets, bare returns.
_NOISE_LINE = re.compile(r"^(//|\*|/\*|import |export \{|[}\])]+;?,?$|\{$|\} else \{$|return;?$)")
_IMPLICIT = {"self", "cls"}
_TEST_PREFIX = "test_"
_DUNDER = "__"

Function = ast.FunctionDef | ast.AsyncFunctionDef


class _Renamer(ast.NodeTransformer):
    """Rename every variable and parameter by order of appearance (``v0``, ``v1``...)."""

    def __init__(self) -> None:
        """Start with no name seen."""
        self.names: dict[str, str] = {}

    def _rename(self, name: str) -> str:
        """Give a name its placeholder.

        Args:
            name: Original name.

        Returns:
            Its placeholder, the same for every use.
        """
        return self.names.setdefault(name, f"v{len(self.names)}")

    def visit_Name(self, node: ast.Name) -> ast.Name:  # pylint: disable=invalid-name  # ast's API
        """Rename a variable.

        Args:
            node: The name.

        Returns:
            The renamed name.
        """
        return ast.copy_location(ast.Name(id=self._rename(node.id), ctx=node.ctx), node)


def _body(function: Function) -> list[ast.stmt]:
    """Take a function's statements, its docstring left out.

    Args:
        function: The function.

    Returns:
        Its statements.
    """
    body = function.body
    first = body[0] if body else None
    is_docstring = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    return body[1:] if is_docstring else body


def _shape(statements: list[ast.stmt]) -> str:
    """Describe statements independently of their variable names.

    Args:
        statements: Statements.

    Returns:
        Their AST dump with variables renamed by order of appearance.
    """
    renamer = _Renamer()
    copies = [renamer.visit(ast.parse(ast.unparse(s)).body[0]) for s in statements]
    return ast.dump(ast.Module(body=copies, type_ignores=[]))


def _is_plain_call(node: ast.expr | None) -> bool:
    """Tell whether an expression is a call whose arguments are all names or constants.

    Args:
        node: Expression (or ``None``).

    Returns:
        Whether it is such a call, or a tuple of such calls.
    """
    if isinstance(node, ast.Tuple):
        return all(_is_plain_call(e) for e in node.elts)
    if not isinstance(node, ast.Call):
        return False
    arguments = [*node.args, *(k.value for k in node.keywords)]
    return all(isinstance(a, (ast.Name, ast.Attribute, ast.Constant)) for a in arguments)


def _is_wiring(statements: list[ast.stmt]) -> bool:
    """Tell whether statements are only plain calls, assigned or not (no logic).

    Args:
        statements: Statements.

    Returns:
        Whether every one is ``x = f(...)``, ``x, y = f(...), g(...)`` or ``f(...)``.
    """
    return all(
        isinstance(s, (ast.Assign, ast.Expr)) and _is_plain_call(s.value) for s in statements
    )


def _functions() -> "list[tuple[Path, Function]]":
    """List every function of the project's Python files.

    Returns:
        ``(file, function)`` pairs.
    """
    return [(path, function) for path, function, _owner in _owned_functions()]


def _owned_functions() -> "list[tuple[Path, Function, str]]":
    """List every function of the project's Python files, with the class it belongs to.

    Returns:
        ``(file, function, class name or "")`` triples.
    """
    found: list[tuple[Path, Function, str]] = []
    for path in python_files():
        tree = ast.parse(path.read_text())
        owners = {
            id(member): node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            for member in node.body
        }
        found.extend(
            (path, node, owners.get(id(node), ""))
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    return found


def _where(path: "Path", node: ast.stmt) -> str:
    """Name a place in the project.

    Args:
        path: File.
        node: Statement or function there.

    Returns:
        ``file:line``, relative to the repository.
    """
    return f"{path.relative_to(ROOT)}:{node.lineno}"


def _repeated(groups: "dict[str, set[str]]") -> list[str]:
    """Keep the groups found in more than one place.

    Args:
        groups: Places by shape.

    Returns:
        One line per repeated shape, listing its places.
    """
    return [" | ".join(sorted(places)) for places in groups.values() if len(places) > 1]


def _is_exempt(function: Function, owner: str) -> bool:
    """Tell whether a function may be a forwarder (it adds behaviour or a required name).

    Args:
        function: The function.
        owner: Its class's name, or ``""``.

    Returns:
        Whether it is decorated, special, a test, or a :data:`PROTOCOL_ADAPTERS` method.
    """
    return bool(
        function.decorator_list
        or function.name.startswith((_DUNDER, _TEST_PREFIX))
        or f"{owner}.{function.name}" in PROTOCOL_ADAPTERS
    )


def _own_parameters(function: Function) -> list[str]:
    """List the parameters a caller passes (``self``/``cls`` and ``*``/``**`` ones left out).

    Args:
        function: The function.

    Returns:
        Their names, in order.
    """
    arguments = function.args
    every = (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    return [a.arg for a in every if a.arg not in _IMPLICIT]


def _passed_through(call: ast.Call) -> list[str]:
    """List the names a call passes on unchanged: positional names, then ``x=x`` keywords.

    Args:
        call: The call.

    Returns:
        Those names, in order.
    """
    positional = [a.id for a in call.args if isinstance(a, ast.Name)]
    # `k.value.id`, equal to `k.arg` here, is typed `str` where `k.arg` is `str | None`.
    keywords = [
        k.value.id for k in call.keywords if isinstance(k.value, ast.Name) and k.value.id == k.arg
    ]
    return [*positional, *keywords]


def _forwarded_call(function: Function) -> ast.Call | None:
    """Find the call a function only forwards its parameters to, unchanged and in order.

    Args:
        function: The function.

    Returns:
        That call, or ``None`` if the function does anything else.
    """
    body = _body(function)
    call = body[0].value if len(body) == 1 and isinstance(body[0], ast.Return) else None
    if not isinstance(call, ast.Call):
        return None
    parameters = _own_parameters(function)
    every_argument_passed = len(call.args) + len(call.keywords) == len(parameters)
    forwards = parameters and _passed_through(call) == parameters and every_argument_passed
    return call if forwards else None


def test_no_function_only_forwards_its_arguments() -> None:
    """No function's whole body is a call passing on its own parameters, unchanged."""
    found = [
        f"{_where(path, function)} {function.name} -> {ast.unparse(call)}"
        for path, function, owner in _owned_functions()
        if not _is_exempt(function, owner) and (call := _forwarded_call(function)) is not None
    ]
    assert found == []


def test_no_two_functions_share_a_body() -> None:
    """No two functions do the same thing under different names."""
    groups: dict[str, set[str]] = defaultdict(set)
    for path, function in _functions():
        shape = _shape(_body(function))
        if len(shape) >= MIN_BODY_SIZE:
            groups[shape].add(f"{_where(path, function)} {function.name}")
    assert _repeated(groups) == []


def test_no_logic_is_repeated() -> None:
    """No run of statements with logic appears in two places (wiring aside)."""
    groups: dict[str, set[str]] = defaultdict(set)
    for path, function in _functions():
        body = _body(function)
        for start in range(len(body) - RUN_STATEMENTS + 1):
            run = body[start : start + RUN_STATEMENTS]
            shape = _shape(run)
            if len(shape) >= MIN_RUN_SIZE and not _is_wiring(run):
                groups[shape].add(f"{_where(path, run[0])} {function.name}")
    assert _repeated(groups) == []


def _meaningful_lines(path: "Path") -> list[tuple[int, str]]:
    """List a script's lines that carry logic (see :data:`_NOISE_LINE`).

    Args:
        path: The script.

    Returns:
        ``(line number, stripped text)`` pairs.
    """
    return [
        (number, text)
        for number, raw in enumerate(path.read_text().splitlines(), start=1)
        if (text := raw.strip()) and not _NOISE_LINE.match(text)
    ]


def test_no_script_lines_are_repeated() -> None:
    """No run of meaningful JavaScript lines appears in two places."""
    groups: dict[str, set[str]] = defaultdict(set)
    scripts = [p for p in project_files() if p.suffix in SCRIPT_SUFFIXES]
    for path in scripts:
        lines = _meaningful_lines(path)
        for start in range(len(lines) - RUN_LINES + 1):
            run = [text for _, text in lines[start : start + RUN_LINES]]
            if sum(map(len, run)) >= MIN_LINES_SIZE:
                groups["\n".join(run)].add(f"{path.relative_to(ROOT)}:{lines[start][0]}")
    assert _repeated(groups) == []


def test_the_scans_cover_the_project() -> None:
    """Every source directory is scanned (a renamed one would silently escape the checks)."""
    assert {p.relative_to(ROOT).parts[0] for p in python_files()} == set(SOURCE_DIRS)
