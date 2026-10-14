"""Keep the tables quoted in Markdown documents identical to the generated tables.

A document includes a generated file between two markers::

    <!-- include: results/tables/main_benchmark_results.md -->
    ...replaced by the file's content...
    <!-- end include -->

:func:`sync` rewrites everything between the markers; :func:`check` reports documents that
are out of date, so CI can fail when a number in the docs no longer matches the results.
Paths are relative to the repository root.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

INCLUDE = re.compile(
    r"(?P<open><!-- include: (?P<path>\S+) -->\n)(?P<body>.*?)(?P<close><!-- end include -->)",
    re.DOTALL,
)


def default_documents(root: Path) -> list[Path]:
    return [root / "README.md", *sorted((root / "docs").glob("*.md"))]


def render(text: str, root: Path) -> str:
    """``text`` with every include block replaced by the current content of its file."""

    def replace(match: re.Match[str]) -> str:
        source = root / match["path"]
        if not source.is_file():
            raise FileNotFoundError(f"included file not found: {match['path']}")
        content = source.read_text(encoding="utf-8")
        if not content.endswith("\n"):
            content += "\n"
        return f"{match['open']}{content}{match['close']}"

    return INCLUDE.sub(replace, text)


def stale(documents: Iterable[Path], root: Path) -> list[Path]:
    """Documents whose include blocks differ from the files they include."""
    out_of_date = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        if render(text, root) != text:
            out_of_date.append(document)
    return out_of_date


def sync(documents: Iterable[Path], root: Path) -> list[Path]:
    """Rewrite out-of-date documents; returns the ones that changed."""
    changed = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        rendered = render(text, root)
        if rendered != text:
            document.write_text(rendered, encoding="utf-8", newline="\n")
            changed.append(document)
    return changed
