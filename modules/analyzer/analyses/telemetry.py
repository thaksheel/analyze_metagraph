"""Insufficient miner-side telemetry.

With the default collection, the only miner-emitted on-chain signal is weight setting
(LastUpdate moving for miners without a validator permit); certificates, commitments
and axons are not collected. So Q1 measures that one signal, Q2 measures how much the
validator-side data can tell miners apart, and Q3 asks whether the subnets where miners
are least distinguishable carry more incident burden. The sufficiency analysis (R^2)
is the other half of the evidence for this problem.
"""
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from ..core.stats import components, cosine_matrix, partial_spearman
from .base import Analysis


class TelemetryAnalysis(Analysis):
    key = "telemetry"
    title = "Insufficient miner-side telemetry"
    problem = "insufficient miner-side telemetry"
    hypothesis = "<10% of subnets collect miner-level signals"
    questions = {
        "Q1": "How many miners emit anything on-chain themselves (weight setting)?",
        "Q2": "How distinguishable are miners using only validator-side data?",
        "Q3": "Do subnets where miners are least distinguishable suffer more incident burden?",
    }
    tables = {
        "subnets": ("Q1-Q2", "per subnet: weight-setting miners, indistinguishable miners"),
        "q1_summary": ("Q1", "share of subnets where miners set weights"),
        "q2_summary": ("Q2", "typical share of indistinguishable miners"),
        "burden": ("Q3", "per subnet: observability vs exploit at-risk and sybil excess"),
        "correlations": ("Q3", "Spearman (and size-controlled) correlations of observability with burden"),
        "q3_summary": ("Q3", "burden in the least vs most observable half, with a significance test"),
    }

    def per_subnet(self, sn, panel, ctx):
        c, g = self.cfg.telemetry, self.cfg.general
        setters = float(panel["miner_weight_setter_share"].mean())
        W = sn.snaps(c.window_days)
        dups, distinct = [], []
        for a in range(0, sn.T, W):
            b = min(sn.T, a + W)
            if b - a < max(10, W // 2):
                continue
            stable = sn.is_miner[a:b].all(axis=0) & ~sn.replaced[a + 1:b].any(axis=0)
            share = sn.miner_emi_share[a:b]
            uids = np.where(stable & (share.mean(axis=0) >= g.min_miner_share_factor / max(int(stable.sum()), 1)))[0]
            if len(uids) < 3:
                continue
            traj, prof = sn.miner_profiles(a, b, uids)
            unit = []
            for p in [traj] + ([prof] if prof is not None else []):
                p = p - p.mean(axis=1, keepdims=True)
                n = np.linalg.norm(p, axis=1, keepdims=True)
                unit.append(p / np.where(n == 0, 1, n))
            S = cosine_matrix(np.hstack(unit) / np.sqrt(len(unit)))
            np.fill_diagonal(S, -1)
            dups.append(float((S.max(axis=1) >= c.duplicate_similarity).mean()))
            distinct.append(len(components(S >= c.duplicate_similarity, 1)) / len(uids))
        ind = float(np.mean(dups)) if dups else np.nan
        return {"subnets": pd.DataFrame([{
            "netuid": sn.netuid, "miner_weight_setter_share": setters,
            "miners_set_weights": bool(setters > 0.01),
            "indistinguishable_share": ind, "distinct_profile_ratio": float(np.mean(distinct)) if distinct else np.nan,
            "observability": 1 - ind if np.isfinite(ind) else np.nan}])}

    def aggregate(self, results, panel):
        s = self.combined("subnets")
        out = {"q1_summary": pd.DataFrame([{
                   "subnets": len(s), "share_subnets_where_miners_set_weights": s["miners_set_weights"].mean(),
                   "median_weight_setter_share": s["miner_weight_setter_share"].median(),
                   "note": "certificates / commitments / axons not in the default collection"}]),
               "q2_summary": pd.DataFrame([{
                   "median_indistinguishable_share": s["indistinguishable_share"].median(),
                   "p90_indistinguishable_share": s["indistinguishable_share"].quantile(0.9),
                   "median_distinct_profile_ratio": s["distinct_profile_ratio"].median()}])}
        m = s.merge(panel.groupby("netuid")["n_uids"].mean().rename("n_uids"), on="netuid", how="left")
        inc = results.get("exploits_incidents")
        ex = results.get("exploits_subnets")
        m = m.merge(ex[["netuid", "n_incidents"]], on="netuid", how="left") if len(ex) else m.assign(n_incidents=0)
        if len(inc):
            m = m.merge(inc.groupby("netuid").agg(at_risk_subnet_share=("at_risk_subnet_share", "sum"),
                                                  median_recovery_hours=("recovery_hours", "median")),
                        on="netuid", how="left")
        else:
            m["at_risk_subnet_share"], m["median_recovery_hours"] = 0.0, np.nan
        sy = results.get("sybil_subnets")
        if len(sy):
            sy = sy.assign(sybil_excess_subnet_share=sy["sybil_excess_alpha"] / sy["subnet_alpha_total"].replace(0, np.nan))
            m = m.merge(sy[["netuid", "sybil_member_share_mean", "sybil_excess_subnet_share"]], on="netuid", how="left")
        else:
            m["sybil_member_share_mean"], m["sybil_excess_subnet_share"] = np.nan, 0.0
        for col in ("at_risk_subnet_share", "sybil_excess_subnet_share"):
            m[col] = m[col].fillna(0.0)
        m["incident_burden"] = m["at_risk_subnet_share"] + m["sybil_excess_subnet_share"]
        out["burden"] = m

        rows = []
        for x in ("observability", "miner_weight_setter_share"):
            for y in ("n_incidents", "median_recovery_hours", "at_risk_subnet_share",
                      "sybil_member_share_mean", "incident_burden"):
                xv, yv = m[x].to_numpy(float), m[y].to_numpy(float)
                ok = np.isfinite(xv) & np.isfinite(yv)
                r = (stats.spearmanr(xv[ok], yv[ok]) if ok.sum() >= 5 and np.ptp(xv[ok]) > 0 and np.ptp(yv[ok]) > 0
                     else None)
                pr, pp = partial_spearman(xv, yv, m["n_uids"].to_numpy(float))
                rows.append({"signal": x, "outcome": y, "n": int(ok.sum()),
                             "spearman_r": float(r.statistic) if r else np.nan, "p_value": float(r.pvalue) if r else np.nan,
                             "partial_r_ctrl_size": pr, "partial_p": pp})
        out["correlations"] = pd.DataFrame(rows)

        ob = m["observability"]
        low, high = (m[ob <= ob.median()], m[ob > ob.median()]) if ob.nunique() > 1 else (m, m.iloc[0:0])
        gap = float(low["incident_burden"].mean() - high["incident_burden"].mean()) if len(low) and len(high) else np.nan
        p = (float(stats.mannwhitneyu(low["incident_burden"], high["incident_burden"], alternative="greater").pvalue)
             if len(low) >= 3 and len(high) >= 3 else np.nan)
        out["q3_summary"] = pd.DataFrame([{
            "least_observable_subnets": len(low), "most_observable_subnets": len(high),
            "burden_least_observable": low["incident_burden"].mean() if len(low) else np.nan,
            "burden_most_observable": high["incident_burden"].mean() if len(high) else np.nan,
            "excess_burden": gap, "p_value": p,
            "evidence_supported": bool(np.isfinite(p) and p < 0.1 and min(len(low), len(high)) >= 5)}])
        return out

    def rank_entry(self, results) -> Dict:
        """Size: share of subnets where miners emit nothing on-chain. Impact: extra burden in the
        least observable half, counted only when the difference is significant."""
        s = results.get("telemetry_subnets").set_index("netuid")
        q3 = results.get("telemetry_q3_summary").iloc[0]
        b = results.get("telemetry_burden").set_index("netuid")
        gap = max(float(q3["excess_burden"]), 0.0) if bool(q3["evidence_supported"]) and np.isfinite(q3["excess_burden"]) else 0.0
        low = b.index[b["observability"] <= b["observability"].median()]
        tot = results.get("exploits_subnets").set_index("netuid")
        v = results.get("sufficiency_verdict")
        suff = ("; R² " + ", ".join(f"{t} {r:.2f}" for t, r, w in zip(v["target"], v["best_r2_all_signals"], v["verdict"])
                                   if not str(w).startswith("not used"))) if len(v) else ""
        return {"size_metric": "share of subnets where miners emit no on-chain signal",
                "size_value": 1 - float(s["miners_set_weights"].mean()),
                "affected_alpha": pd.Series(gap, index=low) * tot["subnet_alpha_flow_total"].reindex(low),
                "affected_tao": pd.Series(gap, index=low) * tot["subnet_tao_flow_total"].reindex(low),
                "measured": f"median indistinguishable miners {results.get('telemetry_q2_summary').iloc[0]['median_indistinguishable_share']:.0%}; "
                            f"burden gap {q3['excess_burden']:.2%} (p={q3['p_value']:.2g}, "
                            f"{'counted' if gap > 0 else 'not counted'}){suff}"}
