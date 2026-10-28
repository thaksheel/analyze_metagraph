import os, time
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
from .analyses import ExploitAnalysis, Analysis
from .config import ExploitConfig, Config

DEPENDS = {
    "telemetry": ["exploits", "sybil"],
    "insights": ["exploits", "sybil"],
    "sufficiency": ["sybil", "rewards"],
}


class Analyzer:
    def __init__(
        self,
        blocks: Optional[str] = None,
        config: Optional[Config] = None,
        out_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.config = config or Config()
        if blocks:
            self.config.paths.blocks = blocks
        if out_dir:
            self.config.paths.cache_dir = os.path.join(out_dir, "cache")
            self.config.paths.results_dir = os.path.join(out_dir, "results")
        self.verbose = verbose
        self._ds: Optional[Dataset] = None
        self.results: Optional[Results] = None
        self.panel: Optional[pd.DataFrame] = None

    # ---------------------------------------------------------------- data
    @property
    def dataset(self) -> Dataset:
        if self._ds is None:
            self._ds = Dataset(self.config, verbose=self.verbose)
        return self._ds

    def subnet(self, netuid: int) -> Subnet:
        return self.dataset[netuid]

    def subnet_panel(self, netuid: int) -> pd.DataFrame:
        if self.panel is not None:
            return self.panel[self.panel.netuid == netuid]
        return PanelBuilder(self.config).build(self.subnet(netuid))[0]

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
            os.path.join(self.config.paths.results_dir, "_catalog.csv")
        ):
            res = self.results = Results.load(
                self.config.paths.results_dir
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
        c = self.config.search(term)
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
    def _collect_analyzers(self, keys: Optional[List[str]]) -> List[Analysis]:
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
        return [a(self.config) for a in ALL_ANALYSES if a.key in want]

    def run_exploits(self, exploit_config: ExploitConfig, save: bool = True) -> Results:
        self.config.exploits = exploit_config
        builder = PanelBuilder(self.config)
        results = Results()
        results.meta = self.dataset.meta
        exploit_analyzer = ExploitAnalysis(self.config)
        panels = []
        for i, sn in enumerate(self.dataset):
            panel, flags = builder.build(sn)
            panels.append(panel)
            context: Dict = {"flags": flags}
            df_objs = exploit_analyzer.per_subnet(sn, panel, context)
            exploit_analyzer.collect(df_objs)
        self.panel = pd.concat(panels, ignore_index=True)
        exploit_analyzer.publish_per_subnet(results)
        exploit_analyzer.publish(results, exploit_analyzer.aggregate(results, self.panel))
        results.add(
            "panel",
            "snapshots",
            self.panel,
            "",
            "one row per subnet and snapshot: every derived metric",
        )
        if save:
            results.save(self.config.paths.results_dir)
            write_report(results, self.config.paths.results_dir, [o for o in objs])
            self._log(
                f"saved {len(results.catalog())} tables to {self.config.paths.results_dir} in {time.time() - t0:.0f}s"
            )
        return results

    def run_all(
        self, analyzers_names: Optional[List[str]] = None, save: bool = True
    ) -> Results:
        t0 = time.time()
        analyzers = self._collect_analyzers(analyzers_names)
        results = Results()
        results.meta = self.dataset.meta
        builder = PanelBuilder(self.config)
        panels = []
        self._log(
            f"running {[o.key for o in analyzers]} on {len(self.dataset)} subnets"
        )
        for i, sn in enumerate(self.dataset):
            panel, flags = builder.build(sn)
            panels.append(panel)
            ctx: Dict = {"flags": flags}
            for analyzer in analyzers:
                out = analyzer.per_subnet(sn, panel, ctx)
                analyzer.collect(out)
                if analyzer.key == "sybil":
                    ctx["sybil_members"] = out.get("members", pd.DataFrame())
                if analyzer.key == "rewards" and len(out.get("subnets", [])):
                    ctx["mechanism_type"] = out["subnets"]["mechanism_type"].iloc[0]
            self._log(f"  [{i + 1}/{len(self.dataset)}] netuid {sn.netuid}")
        self.panel = pd.concat(panels, ignore_index=True)
        results.add(
            "panel",
            "snapshots",
            self.panel,
            "",
            "one row per subnet and snapshot: every derived metric",
        )
        for analyzer in analyzers:
            analyzer.publish_per_subnet(results)
            analyzer.publish(results, analyzer.aggregate(results, self.panel))
            self._log(f"  aggregated {analyzer.key}")
        if all(p in analyzers for p in PROBLEM_ANALYSES):
            probs = [o for o in analyzers if type(o) in PROBLEM_ANALYSES]
            results.add(
                "ranking",
                "problems",
                Ranking(self.config).build(results, probs),
                "",
                "the five problems ranked by size x impact",
            )
        self.results = results
        if save:
            results.save(self.config.paths.results_dir)
            write_report(results, self.config.paths.results_dir, [o for o in analyzers])
            self._log(
                f"saved {len(results.catalog())} tables to {self.config.paths.results_dir} in {time.time() - t0:.0f}s"
            )
        return results

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)
