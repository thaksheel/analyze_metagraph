"""Area 4 — Need for actionable metagraph insights.

Q1  What is the effective dimensionality of the available signals?
Q2  Which signals lead incidents?
Q3  What does the detection gap cost?

Incidents (ground truth) come from Area 1 change points and, when registration
blocks are available, Area 5 cluster onsets.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .config import SECONDS_PER_BLOCK, Config
from .features import SIGNAL_COLUMNS
from .metrics import auc, trailing_z

TOP_SIGNALS = 3


def _snaps(panel_sub: pd.DataFrame, days: float) -> int:
    b = panel_sub["block"].to_numpy()
    spacing = float(np.median(np.diff(b))) if len(b) > 1 else 1.0
    return max(1, int(round(days * 86400 / (spacing * SECONDS_PER_BLOCK))))


def _incident_list(R) -> pd.DataFrame:
    rows = []
    inc = R.get("a1_incidents", pd.DataFrame())
    for _, r in inc.iterrows():
        rows.append({"netuid": r["netuid"], "t": int(r["t"]), "source": "regime_shift",
                     "incident_id": r["incident_id"]})
    clu = R.get("a5_clusters", pd.DataFrame())
    if len(clu) and "onset_t" in clu:
        for _, r in clu.dropna(subset=["onset_t"]).iterrows():
            rows.append({"netuid": r["netuid"], "t": int(r["onset_t"]), "source": "sybil_onset",
                         "incident_id": f"{r['netuid']}-c{r['window']}-{r['cluster']}"})
    return pd.DataFrame(rows).drop_duplicates(subset=["netuid", "t"]) if rows else pd.DataFrame(
        columns=["netuid", "t", "source", "incident_id"])


def q1_dimensionality(panel: pd.DataFrame, meta: dict, cfg: Config) -> Dict[str, pd.DataFrame]:
    cols = [c for c in SIGNAL_COLUMNS if c in panel and panel[c].notna().any()]
    Z = panel[["netuid"] + cols].copy()
    for c in cols:  # standardize within subnet so subnet size does not dominate
        g = Z.groupby("netuid")[c]
        sd = g.transform("std").replace(0, np.nan)
        Z[c] = ((Z[c] - g.transform("mean")) / sd)
    X = Z[cols]
    keep = [c for c in cols if X[c].notna().mean() > 0.5 and X[c].std() > 0]
    X = X[keep].fillna(0.0).to_numpy()
    X = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    cum = np.cumsum(ev)
    n80 = int(np.searchsorted(cum, 0.8) + 1)
    n90 = int(np.searchsorted(cum, cfg.variance_target) + 1)

    C = np.corrcoef(X.T)
    D = 1 - np.abs(np.nan_to_num(C))
    np.fill_diagonal(D, 0)
    labels = fcluster(linkage(squareform(D, checks=False), "average"), t=0.2, criterion="distance")
    groups = pd.DataFrame({"signal": keep, "redundancy_group": labels})

    raw_fields = [f for f in meta.get("fields_present", []) if f not in ("block_info", "netuid")]
    loadings = pd.DataFrame(Vt[:min(n90, len(keep))].T, index=keep,
                            columns=[f"PC{i + 1}" for i in range(min(n90, len(keep)))])
    summary = pd.DataFrame([{
        "raw_storage_fields": len(raw_fields),
        "derived_signals_used": len(keep),
        "components_80pct": n80,
        f"components_{int(cfg.variance_target * 100)}pct": n90,
        "redundancy_groups_r_0_8": int(groups["redundancy_group"].nunique()),
        "redundancy_ratio": 1 - n90 / max(len(keep), 1),
    }])
    ev_df = pd.DataFrame({"component": np.arange(1, len(ev) + 1), "explained": ev, "cumulative": cum})
    return {"a4_q1_summary": summary, "a4_q1_explained_variance": ev_df,
            "a4_q1_redundancy_groups": groups, "a4_q1_loadings": loadings.reset_index(names="signal")}


def _zscores(panel: pd.DataFrame, cols: List[str], cfg: Config) -> Dict[int, pd.DataFrame]:
    out = {}
    for net, p in panel.groupby("netuid"):
        p = p.sort_values("t")
        w = _snaps(p, cfg.baseline_days)
        out[net] = pd.DataFrame({c: np.abs(trailing_z(p[c].to_numpy(float), w)) for c in cols},
                                index=p["t"].to_numpy())
    return out


def q2_leading(panel: pd.DataFrame, incidents: pd.DataFrame, cfg: Config):
    cols = [c for c in SIGNAL_COLUMNS if c in panel and panel[c].notna().any()]
    Z = _zscores(panel, cols, cfg)
    hours = {net: p.set_index("t")["hours"] for net, p in panel.groupby("netuid")}
    scores = {c: [] for c in cols}
    labels = []
    lead_rows = []
    for net, z in Z.items():
        L = _snaps(panel[panel.netuid == net], cfg.lookback_days)
        T = len(z)
        lab = np.zeros(T, bool)
        excl = np.zeros(T, bool)
        sub_inc = incidents[incidents.netuid == net]
        for _, r in sub_inc.iterrows():
            t0 = int(r["t"])
            lab[max(0, t0 - L):t0 + 1] = True
            excl[t0 + 1:min(T, t0 + L + 1)] = True
        use = ~excl | lab
        labels.append(lab[use])
        for c in cols:
            scores[c].append(z[c].to_numpy()[use])
        h = hours[net].to_numpy()
        for _, r in sub_inc.iterrows():
            t0 = int(r["t"])
            for c in cols:
                seg = z[c].to_numpy()[max(0, t0 - L):t0 + 1]
                hit = np.where(seg >= cfg.signal_z)[0]
                lead_rows.append({"netuid": net, "incident_id": r["incident_id"], "signal": c,
                                  "hit": bool(len(hit)),
                                  "lead_hours": float(h[t0] - h[max(0, t0 - L) + hit[0]]) if len(hit) else np.nan})
    y = np.concatenate(labels) if labels else np.array([], bool)
    leads = pd.DataFrame(lead_rows)
    rows = []
    for c in cols:
        s = np.concatenate(scores[c]) if scores[c] else np.array([])
        neg = s[~y] if len(s) else s
        fa = float(np.nanmean(neg >= cfg.signal_z)) if len(neg) else np.nan
        lc = leads[leads.signal == c] if len(leads) else leads
        rows.append({"signal": c, "auc": auc(s, y) if len(s) else np.nan,
                     "hit_rate": float(lc["hit"].mean()) if len(lc) else np.nan,
                     "median_lead_hours": float(lc["lead_hours"].median()) if len(lc) else np.nan,
                     "false_alarm_rate": fa})
    res = pd.DataFrame(rows).sort_values("auc", ascending=False)
    res["actionable"] = (res["auc"] >= 0.65) & (res["hit_rate"] >= 0.3)
    return res, Z


def q3_gap(panel: pd.DataFrame, a1_inc: pd.DataFrame, ranked: pd.DataFrame, Z, cfg: Config) -> pd.DataFrame:
    top = ranked.dropna(subset=["auc"]).head(TOP_SIGNALS)["signal"].tolist()
    rows = []
    if not len(a1_inc) or not top:
        return pd.DataFrame(rows)
    for _, r in a1_inc.iterrows():
        net, t0 = r["netuid"], int(r["t"])
        p = panel[panel.netuid == net].set_index("t").sort_index()
        z = Z[net][top].max(axis=1).to_numpy()
        L = _snaps(p.reset_index(), cfg.lookback_days)
        resp = r["response_t"]
        # detection = first firing once the incident has started (leads are Q2's job;
        # counting pre-incident firings here would mostly count false alarms)
        end = int(resp) if np.isfinite(resp) else min(len(z) - 1, t0 + 2 * L)
        win = np.where(z[t0:end + 1] >= cfg.signal_z)[0]
        t_det = t0 + int(win[0]) if len(win) else np.nan
        rec = {"netuid": net, "incident_id": r["incident_id"], "t": t0,
               "detected": bool(len(win)), "detect_t": t_det,
               "responded": bool(np.isfinite(resp)), "response_t": resp}
        if len(win) and np.isfinite(resp) and resp > t_det:
            seg = p.loc[int(t_det):int(resp) - 1]
            rec["gap_hours"] = float(p.loc[int(resp), "hours"] - p.loc[int(t_det), "hours"])
            rec["gap_cost_tao"] = float(r["cohort_excess_share"] * seg["miner_tao_flow"].sum())
            rec["gap_cost_alpha"] = float(r["cohort_excess_share"] * seg["miner_alpha_flow"].sum())
        else:
            rec["gap_hours"] = 0.0 if len(win) and np.isfinite(resp) else np.nan
            rec["gap_cost_tao"] = rec["gap_cost_alpha"] = 0.0 if rec["gap_hours"] == 0.0 else np.nan
        # incidents nobody responded to: the whole at-risk amount is an upper bound
        rec["unresponded_at_risk_tao"] = 0.0 if np.isfinite(resp) else float(r["at_risk_tao"])
        rec["unresponded_at_risk_alpha"] = 0.0 if np.isfinite(resp) else float(r["at_risk_alpha"])
        rows.append(rec)
    return pd.DataFrame(rows)


def run(panel: pd.DataFrame, R: Dict[str, pd.DataFrame], meta: dict, cfg: Config) -> Dict[str, pd.DataFrame]:
    out = q1_dimensionality(panel, meta, cfg)
    incidents = _incident_list(R)
    ranked, Z = q2_leading(panel, incidents, cfg)
    out["a4_q2_signals"] = ranked
    gap = q3_gap(panel, R.get("a1_incidents", pd.DataFrame()), ranked, Z, cfg)
    out["a4_q3_incidents"] = gap
    if len(gap):
        out["a4_q3_summary"] = pd.DataFrame([{
            "incidents": len(gap),
            "share_detected": gap["detected"].mean(),
            "share_responded": gap["responded"].mean(),
            "median_gap_hours": gap["gap_hours"].median(),
            "total_gap_cost_tao": gap["gap_cost_tao"].sum(),
            "total_unresponded_at_risk_tao": gap["unresponded_at_risk_tao"].sum(),
            "top_signals": ", ".join(ranked.dropna(subset=["auc"]).head(TOP_SIGNALS)["signal"]),
        }])
    out["a4_q2_summary"] = pd.DataFrame([{
        "incidents_used": len(incidents),
        "signals_tested": len(ranked),
        "actionable_signals": int(ranked["actionable"].sum()),
        "actionable_list": ", ".join(ranked.loc[ranked["actionable"], "signal"]),
    }])
    return out
