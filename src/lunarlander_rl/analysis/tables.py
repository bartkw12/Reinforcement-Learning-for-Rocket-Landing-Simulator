"""Markdown tables."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

Formatter = Callable[[pd.Series], str]


def markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, Formatter]],
    *,
    align: Sequence[str] | None = None,
) -> str:
    """Render ``frame`` as a GitHub-flavoured Markdown table.

    ``columns`` pairs each header with a function that formats one row, so a cell may
    combine several columns (a mean and its confidence interval, say).
    """
    headers = [header for header, _ in columns]
    align = align or ["left"] + ["right"] * (len(columns) - 1)
    markers = {"left": ":---", "right": "---:", "center": ":---:"}
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(markers[a] for a in align) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row) for _, fmt in columns) + " |")
    return "\n".join(lines) + "\n"


def number(value: Any, digits: int = 1) -> str:
    """Format a number, or "n/a" if it is missing."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def percent(value: Any, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * value:.{digits}f}%"


def with_interval(value: Any, low: Any, high: Any, fmt: Callable[[Any], str] = number) -> str:
    """``value [low, high]``, the form used for every interval in the reports."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{fmt(value)} [{fmt(low)}, {fmt(high)}]"
