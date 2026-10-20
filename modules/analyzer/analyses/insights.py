"""Need for actionable metagraph insights: how many independent signals are there, which ones
lead incidents, and what does detecting late cost?"""
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..config import SECONDS_PER_BLOCK
from ..core.panel import SIGNALS
from ..core.stats import auc, trailing_z
from .base import Analysis


def _snaps(p: pd.DataFrame, days: float) -> int:
    b = p["block"].to_numpy()
    spacing = float(np.median(np.diff(b))) if len(b) > 1 else 1.0
    return max(1, int(round(days * 86400 / (spacing * SECONDS_PER_BLOCK))))


class InsightAnalysis(Analysis):
    key = "insights"
    title = "Need for actionable metagraph insights"
    problem = "need for actionable metagraph insights"
    hypothesis = ">50 raw signals, few actionable"
    questions = {
        "Q1": "What is the effective dimensionality of the available signals?",
        "Q2": "Which signals lead incidents?",
        "Q3": "What does the detection gap cost?",
    }
    tables = {
        "q1_summary": ("Q1", "raw fields, derived signals, components for 80% / 90% of variance"),
        "explained_variance": ("Q1", "PCA explained variance per component"),
        "redundancy_groups": ("Q1", "signals that are near-duplicates (|r| >= 0.8) share a group"),
        "loadings": ("Q1", "PCA loadings of each signal"),
        "incidents_used": ("Q2", "incidents used as ground truth (exploit incidents + sybil-like track onsets)"),
        "signals": ("Q2", "per signal: AUC before incidents, hit rate, median lead, false-alarm rate, actionable"),
        "q2_summary": ("Q2", "number of actionable signals"),
        "gap_incidents": ("Q3", "per incident: detection time, response time, gap, cost"),
        "q3_summary": ("Q3", "share detected, median gap, total gap cost"),
    }

    def _incidents(self, results) -> pd.DataFrame:
        rows = []
        inc = results.get("exploits_incidents")
        for _, r in inc.iterrows():
            rows.append({"netuid": r["netuid"], "t": int(r["t"]), "source": "exploit", "incident_id": r["incident_id"]})
        tr, cm = results.get("sybil_tracks"), results.get("sybil_communities")
        if len(tr) and len(cm):
            first = cm.groupby("track_id")["t_start"].min()
            for _, r in tr[tr["sybil_like"]].iterrows():
                rows.append({"netuid": r["netuid"], "t": int(first[r["track_id"]]), "source": "sybil_onset",
                             "incident_id": r["track_id"]})
        df = pd.DataFrame(rows, columns=["netuid", "t", "source", "incident_id"])
        return df.drop_duplicates(subset=["netuid", "t"])

    def dimensionality(self, panel, meta) -> Dict[str, pd.DataFrame]:
        c = self.cfg.insights
        cols = [s for s in SIGNALS if s in panel and panel[s].notna().any()]
        Z = panel[["netuid"] + cols].copy()
        for s in cols:  # within-subnet standardization
            g = Z.groupby("netuid")[s]
            Z[s] = (Z[s] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        keep = [s for s in cols if Z[s].notna().mean() > 0.5 and Z[s].std() > 0]
        X = Z[keep].fillna(0.0).to_numpy()
        X = X - X.mean(axis=0)
        _, sv, Vt = np.linalg.svd(X, full_matrices=False)
        ev = sv ** 2 / (sv ** 2).sum()
        cum = np.cumsum(ev)
        n80, n90 = int(np.searchsorted(cum, 0.8) + 1), int(np.searchsorted(cum, c.variance_target) + 1)
        D = 1 - np.abs(np.nan_to_num(np.corrcoef(X.T)))
        np.fill_diagonal(D, 0)
        labels = fcluster(linkage(squareform(D, checks=False), "average"), t=0.2, criterion="distance")
        k = min(n90, len(keep))
        return {"q1_summary": pd.DataFrame([{
                    "raw_storage_fields": len(meta.get("fields_present", [])), "derived_signals_used": len(keep),
                    "components_80pct": n80, f"components_{int(c.variance_target * 100)}pct": n90,
                    "redundancy_groups": int(len(set(labels))), "redundancy_ratio": 1 - n90 / max(len(keep), 1)}]),
                "explained_variance": pd.DataFrame({"component": np.arange(1, len(ev) + 1), "explained": ev,
                                                    "cumulative": cum}),
                "redundancy_groups": pd.DataFrame({"signal": keep, "group": labels}),
                "loadings": pd.DataFrame(Vt[:k].T, index=keep, columns=[f"PC{i + 1}" for i in range(k)])
                            .reset_index(names="signal")}

    def _zscores(self, panel, cols: List[str]) -> Dict[int, pd.DataFrame]:
        out = {}
        for net, p in panel.groupby("netuid"):
            p = p.sort_values("t")
            w = _snaps(p, self.cfg.insights.baseline_days)
            out[net] = pd.DataFrame({s: np.abs(trailing_z(p[s].to_numpy(float), w)) for s in cols}, index=p["t"].to_numpy())
        return out

    def leading(self, panel, incidents):
        c = self.cfg.insights
        cols = [s for s in SIGNALS if s in panel and panel[s].notna().any()]
        Z = self._zscores(panel, cols)
        hours = {net: p.set_index("t")["hours"].to_numpy() for net, p in panel.groupby("netuid")}
        scores, labels, leads = {s: [] for s in cols}, [], []
        for net, z in Z.items():
            L = _snaps(panel[panel.netuid == net], c.lookback_days)
            T = len(z)
            lab, excl = np.zeros(T, bool), np.zeros(T, bool)
            sub = incidents[incidents.netuid == net]
            for t0 in sub["t"].astype(int):
                lab[max(0, t0 - L):t0 + 1] = True
                excl[t0 + 1:min(T, t0 + L + 1)] = True
            use = ~excl | lab
            labels.append(lab[use])
            for s in cols:
                scores[s].append(z[s].to_numpy()[use])
            for _, r in sub.iterrows():
                t0 = int(r["t"])
                for s in cols:
                    hit = np.where(z[s].to_numpy()[max(0, t0 - L):t0 + 1] >= c.signal_z)[0]
                    leads.append({"signal": s, "hit": bool(len(hit)),
                                  "lead_hours": float(hours[net][t0] - hours[net][max(0, t0 - L) + hit[0]]) if len(hit) else np.nan})
        y = np.concatenate(labels) if labels else np.array([], bool)
        leads = pd.DataFrame(leads)
        rows = []
        for s in cols:
            sc = np.concatenate(scores[s]) if scores[s] else np.array([])
            lc = leads[leads.signal == s] if len(leads) else leads
            rows.append({"signal": s, "auc": auc(sc, y) if len(sc) else np.nan,
                         "hit_rate": float(lc["hit"].mean()) if len(lc) else np.nan,
                         "median_lead_hours": float(lc["lead_hours"].median()) if len(lc) else np.nan,
                         "false_alarm_rate": float(np.nanmean(sc[~y] >= c.signal_z)) if len(sc) else np.nan})
        res = pd.DataFrame(rows).sort_values("auc", ascending=False)
        res["actionable"] = (res["auc"] >= 0.65) & (res["hit_rate"] >= 0.3)
        return res, Z

    def detection_gap(self, panel, inc, ranked, Z) -> pd.DataFrame:
        c = self.cfg.insights
        top = ranked.dropna(subset=["auc"]).head(c.top_signals)["signal"].tolist()
        rows = []
        for _, r in (inc.iterrows() if len(inc) and top else []):
            net, t0 = r["netuid"], int(r["t"])
            p = panel[panel.netuid == net].set_index("t").sort_index()
            z = Z[net][top].max(axis=1).to_numpy()
            L = _snaps(p.reset_index(), c.lookback_days)
            resp = r["response_t"]
            end = int(resp) if np.isfinite(resp) else min(len(z) - 1, t0 + 2 * L)
            win = np.where(z[t0:end + 1] >= c.signal_z)[0]
            t_det = t0 + int(win[0]) if len(win) else np.nan
            rec = {"netuid": net, "incident_id": r["incident_id"], "t": t0, "detected": bool(len(win)),
                   "detect_t": t_det, "responded": bool(np.isfinite(resp)), "response_t": resp,
                   "gap_hours": np.nan, "gap_cost_alpha": np.nan, "gap_cost_tao": np.nan}
            if len(win) and np.isfinite(resp):
                if resp > t_det:
                    seg = p.loc[int(t_det):int(resp) - 1]
                    rec["gap_hours"] = float(p.loc[int(resp), "hours"] - p.loc[int(t_det), "hours"])
                    rec["gap_cost_alpha"] = float(r["cohort_excess_share"] * seg["miner_alpha_flow"].sum())
                    rec["gap_cost_tao"] = float(r["cohort_excess_share"] * seg["miner_tao_flow"].sum())
                else:
                    rec["gap_hours"] = rec["gap_cost_alpha"] = rec["gap_cost_tao"] = 0.0
            rec["unresponded_at_risk_alpha"] = 0.0 if np.isfinite(resp) else float(r["at_risk_alpha"])
            rec["unresponded_at_risk_tao"] = 0.0 if np.isfinite(resp) else float(r["at_risk_tao"])
            rows.append(rec)
        return pd.DataFrame(rows)

    def aggregate(self, results, panel):
        out = self.dimensionality(panel, results.meta)
        incidents = self._incidents(results)
        ranked, Z = self.leading(panel, incidents)
        gap = self.detection_gap(panel, results.get("exploits_incidents"), ranked, Z)
        out.update({"incidents_used": incidents, "signals": ranked, "gap_incidents": gap,
                    "q2_summary": pd.DataFrame([{
                        "incidents_used": len(incidents), "signals_tested": len(ranked),
                        "actionable_signals": int(ranked["actionable"].sum()),
                        "actionable_list": ", ".join(ranked.loc[ranked["actionable"], "signal"])}])})
        if len(gap):
            out["q3_summary"] = pd.DataFrame([{
                "incidents": len(gap), "share_detected": gap["detected"].mean(), "share_responded": gap["responded"].mean(),
                "median_gap_hours": gap["gap_hours"].median(), "total_gap_cost_tao": gap["gap_cost_tao"].sum(),
                "total_unresponded_at_risk_tao": gap["unresponded_at_risk_tao"].sum(),
                "top_signals": ", ".join(ranked.dropna(subset=["auc"]).head(self.cfg.insights.top_signals)["signal"])}])
        return out

    def rank_entry(self, results) -> Dict:
        gap = results.get("insights_gap_incidents")
        n = len(results.get("exploits_subnets"))
        helped = gap[(gap["gap_hours"] > 0) | (~gap["responded"])] if len(gap) else gap
        q1, q2 = results.get("insights_q1_summary").iloc[0], results.get("insights_q2_summary").iloc[0]
        g = (lambda col: gap.groupby("netuid")[col].sum() if len(gap) else pd.Series(dtype=float))
        return {"size_metric": "share of subnets where earlier detection was possible",
                "size_value": helped["netuid"].nunique() / max(n, 1) if len(helped) else 0.0,
                "affected_alpha": g("gap_cost_alpha"), "affected_tao": g("gap_cost_tao"),
                "upper_alpha": g("unresponded_at_risk_alpha"), "upper_tao": g("unresponded_at_risk_tao"),
                "measured": f"{int(q1['raw_storage_fields'])} raw fields, {int(q1['derived_signals_used'])} derived signals; "
                            f"{int(q1['components_80pct'])} components explain 80%; "
                            f"{int(q2['actionable_signals'])} actionable"}
