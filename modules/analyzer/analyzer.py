"""Analyzer: the one entry point.

from bt_analysis import Analyzer
az = Analyzer("./exports/blocks/")
res = az.run()                          # everything; or az.run(["exploits", "sybil"])
res.ranking.problems                    # ranked problems
res.sybil.tracks                        # any table: res.<analysis>.<table>
az.questions()                          # every question and the tables that answer it
az.find("churn")                        # search questions, tables, columns and settings
az.subnet(20)                           # raw + derived arrays for one subnet
az.cfg.sybil.edge_min = 0.7; az.run(["sybil"])   # tweak and rerun one analysis
"""

import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .analyses import ALL_ANALYSES, BY_KEY, PROBLEM_ANALYSES
from .config import Config
from .core.panel import PanelBuilder
from .data.dataset import Dataset
from .data.subnet import Subnet
from .ranking import Ranking
from .report import write_report
from .results import Results

DEPENDS = {
    "telemetry": ["exploits", "sybil"],
    "insights": ["exploits", "sybil"],
    "sufficiency": ["sybil", "rewards"],
}


class Analyzer:
    def __init__(
        self,
        blocks: Optional[str] = None,
        cfg: Optional[Config] = None,
        out_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.cfg = cfg or Config()
        if blocks:
            self.cfg.paths.blocks = blocks
        if out_dir:
            self.cfg.paths.cache_dir = os.path.join(out_dir, "cache")
            self.cfg.paths.results_dir = os.path.join(out_dir, "results")
        self.verbose = verbose
        self._ds: Optional[Dataset] = None
        self.results: Optional[Results] = None
        self.panel: Optional[pd.DataFrame] = None

    # ---------------------------------------------------------------- data
    @property
    def dataset(self) -> Dataset:
        if self._ds is None:
            self._ds = Dataset(self.cfg, verbose=self.verbose)
        return self._ds

    def subnet(self, netuid: int) -> Subnet:
        return self.dataset[netuid]

    def subnet_panel(self, netuid: int) -> pd.DataFrame:
        if self.panel is not None:
            return self.panel[self.panel.netuid == netuid]
        return PanelBuilder(self.cfg).build(self.subnet(netuid))[0]

    # ---------------------------------------------------------------- discovery
    @staticmethod
    def questions(analysis: Optional[str] = None) -> pd.DataFrame:
        df = pd.concat([a.describe() for a in ALL_ANALYSES], ignore_index=True)
        return df[df["analysis"] == analysis] if analysis else df

    @staticmethod
    def analyses() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "key": a.key,
                    "class": a.__name__,
                    "title": a.title,
                    "problem": a.problem,
                    "needs": ", ".join(DEPENDS.get(a.key, [])),
                }
                for a in ALL_ANALYSES
            ]
        )

    def find(self, term: str) -> pd.DataFrame:
        """Search questions, result tables (names, descriptions, columns) and settings."""
        t = term.lower()
        q = self.questions()
        q = q[q.apply(lambda r: t in " ".join(map(str, r.values)).lower(), axis=1)]
        out = [
            pd.DataFrame(
                {
                    "kind": "question",
                    "where": q["analysis"] + "." + q["question"],
                    "text": q["text"],
                }
            )
        ]
        res = self.results
        if res is None and os.path.exists(
            os.path.join(self.cfg.paths.results_dir, "_catalog.csv")
        ):
            res = self.results = Results.load(
                self.cfg.paths.results_dir
            )  # earlier run on disk
        if res is not None:
            r = res.search(term)
            out.append(
                pd.DataFrame(
                    {"kind": "table", "where": r["name"], "text": r["description"]}
                )
            )
        else:
            rows = [
                {"kind": "table", "where": f"{a.key}_{s}", "text": d}
                for a in ALL_ANALYSES
                for s, (_, d) in a.tables.items()
                if t in f"{s} {d}".lower()
            ]
            out.append(pd.DataFrame(rows, columns=["kind", "where", "text"]))
        c = self.cfg.search(term)
        out.append(
            pd.DataFrame(
                {
                    "kind": "setting",
                    "where": c["setting"],
                    "text": c["value"].astype(str) + "  " + c["note"],
                }
            )
        )
        return pd.concat(out, ignore_index=True)

    # ---------------------------------------------------------------- run
    def _resolve(self, keys: Optional[List[str]]) -> List:
        if keys is None:
            return list(ALL_ANALYSES)
        want = {"checks"}
        stack = list(keys)
        while stack:
            k = stack.pop()
            if k not in BY_KEY:
                raise KeyError(f"unknown analysis '{k}'; choose from {list(BY_KEY)}")
            if k not in want:
                want.add(k)
                stack += DEPENDS.get(k, [])
        return [a for a in ALL_ANALYSES if a.key in want]

    def run(self, analyses: Optional[List[str]] = None, save: bool = True) -> Results:
        t0 = time.time()
        ds = self.dataset
        classes = self._resolve(analyses)
        objs = [cls(self.cfg) for cls in classes]
        res = Results()
        res.meta = ds.meta
        builder = PanelBuilder(self.cfg)
        panels = []
        self._log(f"running {[o.key for o in objs]} on {len(ds)} subnets")
        for i, sn in enumerate(ds):
            panel, flags = builder.build(sn)
            panels.append(panel)
            ctx: Dict = {"flags": flags}
            for o in objs:
                out = o.per_subnet(sn, panel, ctx)
                o.collect(out)
                if o.key == "sybil":
                    ctx["sybil_members"] = out.get("members", pd.DataFrame())
                if o.key == "rewards" and len(out.get("subnets", [])):
                    ctx["mechanism_type"] = out["subnets"]["mechanism_type"].iloc[0]
            self._log(f"  [{i + 1}/{len(ds)}] netuid {sn.netuid}")
        self.panel = pd.concat(panels, ignore_index=True)
        res.add(
            "panel",
            "snapshots",
            self.panel,
            "",
            "one row per subnet and snapshot: every derived metric",
        )
        for o in objs:
            o.publish_per_subnet(res)
            o.publish(res, o.aggregate(res, self.panel))
            self._log(f"  aggregated {o.key}")
        if all(p in classes for p in PROBLEM_ANALYSES):
            probs = [o for o in objs if type(o) in PROBLEM_ANALYSES]
            res.add(
                "ranking",
                "problems",
                Ranking(self.cfg).build(res, probs),
                "",
                "the five problems ranked by size x impact",
            )
        self.results = res
        if save:
            res.save(self.cfg.paths.results_dir)
            write_report(res, self.cfg.paths.results_dir, [o for o in objs])
            self._log(
                f"saved {len(res.catalog())} tables to {self.cfg.paths.results_dir} in {time.time() - t0:.0f}s"
            )
        return res

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)
