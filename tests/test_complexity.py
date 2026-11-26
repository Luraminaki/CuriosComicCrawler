"""No function or method is more complex than radon grade B (the owner's limit)."""

from radon.complexity import cc_visit
from test_conventions import ROOT, python_files

MAX_COMPLEXITY = 10  # radon grade B upper bound; C and worse must be split


def test_no_block_exceeds_grade_b() -> None:
    """Every function, method and class stays at cyclomatic complexity 10 or below."""
    too_complex = [
        f"{path.relative_to(ROOT)}:{block.lineno} {block.name} {block.complexity}"
        for path in python_files()
        for block in cc_visit(path.read_text(encoding="utf-8"))
        if block.complexity > MAX_COMPLEXITY
    ]
    assert too_complex == []
