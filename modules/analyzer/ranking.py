"""Ranking: size x impact for the five focus-area problems.

Each problem analysis defines its own size and impact in `rank_entry()` (see the class).
Impact is the % of network TAO emission affected over the period (TAO-weighted when
prices exist; otherwise the mean affected share of each subnet's own alpha emission).
Perceived importance and solving ease are not measurable on-chain; add them from your
table as a second axis.
"""
from typing import List

import numpy as np
import pandas as pd

from .config import Config
from .results import Results


def _band(v: float, edges: List[float]) -> float:
    return float(1 + sum(v >= e for e in edges)) if np.isfinite(v) else np.nan


class Ranking:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build(self, results: Results, analyses) -> pd.DataFrame:
        subs = results.get("exploits_subnets").set_index("netuid")
        tot_tao, tot_alpha = subs["subnet_tao_flow_total"], subs["subnet_alpha_flow_total"].replace(0, np.nan)
        use_tao = float(tot_tao.sum()) > 0

        def pct(alpha: pd.Series, tao: pd.Series) -> float:
            if use_tao:
                return float(100 * pd.Series(tao, dtype=float).reindex(subs.index).fillna(0).sum() / tot_tao.sum())
            return float(100 * (pd.Series(alpha, dtype=float).reindex(subs.index).fillna(0) / tot_alpha).mean())

        rows = []
        for a in analyses:
            e = a.rank_entry(results)
            row = {"problem": a.problem, "analysis": a.key, "hypothesis": getattr(a, "hypothesis", ""),
                   "size_metric": e["size_metric"], "size_value": float(e["size_value"]),
                   "impact_pct": pct(e["affected_alpha"], e["affected_tao"]), "measured": e["measured"]}
            if "upper_alpha" in e:
                row["impact_upper_pct"] = pct(e["upper_alpha"], e["upper_tao"])
            rows.append(row)
        df = pd.DataFrame(rows)
        r = self.cfg.ranking
        df["size_score"] = df["size_value"].apply(lambda v: _band(v, r.size_bands))
        df["impact_score"] = df["impact_pct"].apply(lambda v: _band(v, r.impact_bands_pct))
        df["priority"] = df["size_score"] * df["impact_score"]
        df["impact_unit"] = "% of network TAO emission" if use_tao else "% of avg subnet emission (alpha)"
        df["supporting_evidence"] = df["analysis"].map(self._evidence(results)).fillna("")
        df = df.sort_values(["priority", "impact_pct"], ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        return df

    @staticmethod
    def _evidence(results: Results) -> dict:
        ev = {}
        s = results.get("rewards_summary")
        if len(s):
            r = s.iloc[0]
            ev["exploits"] = ev["scoring"] = (
                f"rewards: {r['share_spikes_unsupported']:.0%} of reward spikes lack broad validator support; "
                f"systematic offenders in {r['share_subnets_with_systematic_offenders']:.0%} of subnets")
        v = results.get("sufficiency_verdict")
        if len(v):
            used = v[~v["verdict"].str.startswith("not used")]
            ev["telemetry"] = ev["insights"] = "sufficiency: R² " + ", ".join(
                f"{t} {x:.2f} ({w})" for t, x, w in zip(used["target"], used["best_r2_all_signals"], used["verdict"]))
        return ev
