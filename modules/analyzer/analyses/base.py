"""Base class for analyses.

An analysis declares:
  key        short name used in results (res.<key>.<table>) and CSV names
  title      human title
  problem    the focus-area problem it sizes (for the ranking), or ""
  questions  {"Q1": "question text", ...}
  tables     {short_name: ("Q1", "what one row is / what the table holds")}

and implements per_subnet() (called once per subnet with that subnet's data) and/or
aggregate() (called once with all results so far). Outputs of per_subnet() are
concatenated across subnets automatically.
"""

from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd

from ..config import Config
from ..results import Results


def covers(spec: str, q: str) -> bool:
    """Does a table's question spec ('Q2', 'Q1-Q3') cover question q?"""
    if "-" in spec:
        lo, hi = (int(x.strip()[1:]) for x in spec.split("-"))
        return lo <= int(q[1:]) <= hi
    return spec == q


class Analysis:
    key: str = ""
    title: str = ""
    problem: str = ""
    questions: Dict[str, str] = {}
    tables: Dict[str, Tuple[str, str]] = {}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._parts = defaultdict(list)

    # ---- hooks
    def per_subnet(self, sn, panel: pd.DataFrame, ctx: Dict) -> Dict[str, pd.DataFrame]:
        return {}

    def aggregate(
        self, results: Results, panel: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        return {}

    # ---- plumbing
    def collect(self, out: Dict[str, pd.DataFrame]):
        for k, v in out.items():
            if v is not None and len(v):
                self._parts[k].append(v)

    def combined(self, short: str) -> pd.DataFrame:
        parts = self._parts.get(short, [])
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def publish(self, results: Results, tables: Dict[str, pd.DataFrame]):
        for short, df in tables.items():
            q, desc = self.tables.get(short, ("", ""))
            results.add(self.key, short, df, q, desc)

    def publish_per_subnet(self, results: Results):
        self.publish(
            results, {k: self.combined(k) for k in self._parts if not k.startswith("_")}
        )

    @classmethod
    def describe(cls) -> pd.DataFrame:
        rows = [
            {
                "analysis": cls.key,
                "question": q,
                "text": t,
                "tables": ", ".join(
                    s for s, (qq, _) in cls.tables.items() if covers(qq, q)
                ),
            }
            for q, t in cls.questions.items()
        ]
        return pd.DataFrame(rows)
