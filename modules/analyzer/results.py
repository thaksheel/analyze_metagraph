import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any 

import pandas as pd


@dataclass
class TableInfo:
    name: str
    analysis: str
    short: str
    question: str
    description: str


class AnalysisResults:
    """Attribute access to one analysis's tables: res.sybil.tracks."""

    def __init__(self, key: str, results: "Results"):
        self._key = key
        self._results = results

    def tables(self) -> List[str]:
        return [i.short for i in self._results._info.values() if i.analysis == self._key]

    def __getattr__(self, short: str) -> pd.DataFrame:
        if short.startswith("_"):
            raise AttributeError(short)
        try:
            return self._results[f"{self._key}_{short}"]
        except KeyError:
            raise AttributeError(f"'{self._key}' has no table '{short}'; tables: {self.tables()}") from None

    def __dir__(self):
        return self.tables()

    def __repr__(self):
        c = self._results.catalog()
        c = c[c["analysis"] == self._key][["short", "question", "rows", "description"]]
        return f"<{self._key} results>\n" + c.to_string(index=False)


class Results:
    def __init__(self):
        self._tables: Dict[str, pd.DataFrame] = {}
        self._info: Dict[str, TableInfo] = {}
        self.meta: Dict[str, Any] = {}

    def add(self, analysis: str, short: str, df: pd.DataFrame, question: str = "", description: str = ""):
        name = f"{analysis}_{short}"
        self._tables[name] = df if df is not None else pd.DataFrame()
        self._info[name] = TableInfo(name, analysis, short, question, description)

    # ---- access
    def __getitem__(self, name: str) -> pd.DataFrame:
        name = name.replace(".", "_")
        return self._tables[name]

    def get(self, name: str, default: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        try:
            return self[name]
        except KeyError:
            return pd.DataFrame() if default is None else default

    def __contains__(self, name: str) -> bool:
        return name.replace(".", "_") in self._tables

    def analyses(self) -> List[str]:
        return sorted({i.analysis for i in self._info.values()})

    def __getattr__(self, key: str):
        if key.startswith("_"):
            raise AttributeError(key)
        if key in self.analyses():
            return AnalysisResults(key, self)
        if key in self._tables:
            return self._tables[key]
        raise AttributeError(f"no analysis or table '{key}'; analyses: {self.analyses()}")

    def __dir__(self):
        return self.analyses() + list(self._tables)

    # ---- discovery
    def catalog(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "name": i.name, "analysis": i.analysis, "short": i.short, "question": i.question,
            "rows": len(self._tables[i.name]), "columns": ", ".join(map(str, self._tables[i.name].columns)),
            "description": i.description} for i in self._info.values()])

    def search(self, term: str) -> pd.DataFrame:
        c = self.catalog()
        if c.empty:
            return c
        t = term.lower()
        hit = c.apply(lambda r: t in " ".join(map(str, r.values)).lower(), axis=1)
        return c[hit][["name", "question", "rows", "description"]]

    def __repr__(self):
        return f"Results({len(self._tables)} tables from {self.analyses()})"

    # ---- persistence
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for name, df in self._tables.items():
            df.to_csv(os.path.join(directory, f"{name}.csv"), index=False)
        self.catalog().to_csv(os.path.join(directory, "_catalog.csv"), index=False)

    @classmethod
    def load(cls, directory: str) -> "Results":
        """Reload saved CSVs (uses _catalog.csv for questions / descriptions)."""
        res = cls()
        cat = pd.read_csv(os.path.join(directory, "_catalog.csv"))
        for _, r in cat.iterrows():
            p = os.path.join(directory, f"{r['name']}.csv")
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            res.add(r["analysis"], r["short"], df, str(r.get("question", "") or ""), str(r.get("description", "") or ""))
        return res
