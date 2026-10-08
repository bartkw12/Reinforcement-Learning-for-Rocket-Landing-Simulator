"""Ranking hyperparameter candidates and recording which one was selected.

Tuning variants are labelled ``<family>__<key>=<value>__...`` (see the sweep's ``grid``),
and every candidate of a family is a configuration of the same agent.
"""

from __future__ import annotations

import pandas as pd

from lunarlander_rl.analysis.benchmark import display_name
from lunarlander_rl.analysis.tables import markdown_table, number, percent, with_interval

SCORE = "mean_return"


def family(variant: str) -> str:
    return variant.split("__", 1)[0]


def settings(variant: str) -> str:
    """The part of a label that distinguishes a candidate: ``"lr=0.001, baseline=value"``."""
    parts = variant.split("__")[1:]
    return ", ".join(parts) if parts else "(base)"


def ranked(summary: pd.DataFrame) -> pd.DataFrame:
    """Candidates grouped by family, best first within each family."""
    frame = summary.assign(family=summary["variant"].map(family))
    result: pd.DataFrame = frame.sort_values(
        ["family", SCORE, "variant"], ascending=[True, False, True], ignore_index=True
    )
    return result


def selection(summary: pd.DataFrame) -> pd.DataFrame:
    """The best candidate of each family by mean validation return over seeds."""
    best = ranked(summary).groupby("family", sort=True).head(1)
    result: pd.DataFrame = best[["family", "variant", SCORE, "seeds"]].reset_index(drop=True)
    return result


def table(summary: pd.DataFrame) -> str:
    rows = ranked(summary)
    chosen = set(selection(summary)["variant"])
    sections = []
    for name, group in rows.groupby("family", sort=True):
        sections.append(f"#### {display_name(str(name))}\n\n")
        sections.append(
            markdown_table(
                group,
                [
                    (
                        "Candidate",
                        lambda r: (
                            f"**{settings(r['variant'])}** (selected)"
                            if r["variant"] in chosen
                            else settings(r["variant"])
                        ),
                    ),
                    (
                        "Validation return [95% CI]",
                        lambda r: with_interval(
                            r["mean_return"], r["mean_return_ci_low"], r["mean_return_ci_high"]
                        ),
                    ),
                    (
                        "Seed range",
                        lambda r: (
                            f"{number(r['seed_min_return'])} to {number(r['seed_max_return'])}"
                        ),
                    ),
                    ("Success", lambda r: percent(r["success_rate"])),
                    ("Best checkpoint", lambda r: number(r["best_mean_return"])),
                ],
            )
        )
        sections.append("\n")
    return "".join(sections).rstrip("\n") + "\n"
