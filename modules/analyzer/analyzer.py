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
        t0 = time.time()
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
        exploit_analyzer.publish(
            results, exploit_analyzer.aggregate(results, self.panel)
        )
        results.add(
            "panel",
            "snapshots",
            self.panel,
            "",
            "one row per subnet and snapshot: every derived metric",
        )
        if save:
            results.save(self.config.paths.results_dir)
            write_report(results, self.config.paths.results_dir, [exploit_analyzer])
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

    def exploit_aggregated_report(self, results: Results, save: bool = False):
        inc = results["exploits_incidents"]
        n_subnets = len(results["exploits_subnets"])
        n_snapshots = int(results["panel_snapshots"].groupby("netuid").size().median())
        for c in ["burn_response", "hyperparam_response", "recovered"]:
            inc[c] = inc[c].astype(str).str.lower().eq("true")
        rows = []
        for col in inc.columns:
            s = inc[col]
            if s.dtype == bool:
                rows.append(
                    {
                        "column": col,
                        "count": len(s),
                        "true": s.sum(),
                        "share_true": s.mean(),
                    }
                )
            elif pd.api.types.is_numeric_dtype(s):
                rows.append(
                    {
                        "column": col,
                        "count": s.notna().sum(),
                        "sum": s.sum(),
                        "mean": s.mean(),
                        "median": s.median(),
                        "p10": s.quantile(0.1),
                        "p90": s.quantile(0.9),
                        "min": s.min(),
                        "max": s.max(),
                    }
                )
            else:
                rows.append(
                    {
                        "column": col,
                        "count": s.notna().sum(),
                        "distinct": s.nunique(),
                        "most_common": s.mode().iloc[0] if s.notna().any() else None,
                    }
                )
        columns = pd.DataFrame(rows)
        per_subnet = inc.groupby("netuid").size()
        days_between = (
            inc.sort_values("hours")
            .groupby("netuid")["hours"]
            .apply(lambda h: (h.diff() / 24).median())
        )
        response = pd.Series("no response", index=inc.index)
        response[inc.burn_response & ~inc.hyperparam_response] = "burn only"
        response[~inc.burn_response & inc.hyperparam_response] = "settings only"
        response[inc.burn_response & inc.hyperparam_response] = "burn + settings"
        response_share = response.value_counts(normalize=True)

        headline = pd.DataFrame(
            [
                ("subnets analyzed", n_subnets),
                (
                    "share of subnets with repeated incidents (2+)",
                    (per_subnet >= 2).sum() / n_subnets,
                ),
                ("incidents", len(inc)),
                ("snapshots", n_snapshots),
                (
                    "share of subnets with at least one incident",
                    (per_subnet >= 1).sum() / n_subnets,
                ),
                ("median days between incidents in a subnet", days_between.median()),
                (
                    "median share of miners in the gaining group",
                    inc.cohort_uid_share.median(),
                ),
                ("median miners in the gaining group", inc.cohort_size.median()),
                (
                    "median miners in the subnet",
                    (inc.cohort_size / inc.cohort_uid_share).median(),
                ),
                (
                    "median share of newly registered miners in the group",
                    inc.cohort_new_frac.median(),
                ),
                (
                    "share of incidents with any newly registered miner",
                    (inc.cohort_new_frac > 0).mean(),
                ),
                ("response: burn only", response_share.get("burn only", 0)),
                ("response: settings only", response_share.get("settings only", 0)),
                ("response: burn + settings", response_share.get("burn + settings", 0)),
                ("response: none", response_share.get("no response", 0)),
                ("median hours until owner responds", inc.response_hours.median()),
                ("median hours until the shift unwound", inc.recovery_hours.median()),
                ("share of shifts that never unwound", 1 - inc.recovered.mean()),
                ("total TAO at risk", inc.at_risk_tao.sum()),
            ],
            columns=["insight", "value"],
        )
        if save: 
            with pd.ExcelWriter("./exports/output.xlsx", engine="openpyxl") as writer:
                columns.to_excel(writer, sheet_name="features", index=False)
                headline.to_excel(writer, sheet_name="summ", index=False)
        return columns, headline
