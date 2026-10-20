"""summary.md: the ranking, then each analysis's questions with their summary tables."""
import os
from typing import List

import numpy as np
import pandas as pd

from .analyses.base import covers
from .results import Results


def _md(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df is None or not len(df):
        return "_no rows_\n"
    df = df.head(max_rows)

    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.4g}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n| " + " | ".join("---" for _ in df.columns) + " |\n"
    return head + "\n".join("| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n"


def write_report(res: Results, directory: str, analyses: List) -> str:
    lines = ["# Bittensor problem sizing — results\n",
             "Every table is also a CSV in this folder; `_catalog.csv` lists them with the question each answers.\n"]
    if "ranking_problems" in res:
        lines += ["\n## Ranking\n", _md(res["ranking_problems"].drop(columns=["measured", "supporting_evidence"],
                                                                     errors="ignore"))]
    for a in analyses:
        lines.append(f"\n## {a.title}\n")
        for q, text in a.questions.items():
            # each summary-type table is shown once, under the first question it covers
            shown = [s for s, (qq, _) in a.tables.items()
                     if covers(qq, q) and not any(covers(qq, p) for p in list(a.questions)[:list(a.questions).index(q)])
                     and ("summary" in s or s in ("issues", "verdict", "layer_evidence", "stability",
                                                  "by_mechanism", "signals", "mechanism_tests", "yc3"))]
            lines.append(f"\n**{q}. {text}**\n")
            for s in shown:
                name = f"{a.key}_{s}"
                if name in res:
                    lines += [f"\n`{name}`\n\n", _md(res[name])]
    path = os.path.join(directory, "summary.md")
    os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
