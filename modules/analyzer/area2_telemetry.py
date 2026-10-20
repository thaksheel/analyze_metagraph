"""Area 2 — Insufficient miner-side telemetry.

Q1  How many miners emit anything on-chain themselves?
Q2  How distinguishable are miners using only validator-side data?
Q3  Do subnets with less telemetry suffer more or longer incidents?

On-chain only: the chain cannot see off-chain telemetry a subnet collects, so pair
Q1 with a review of each subnet's validator code before testing the <10% claim.
"""
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from .area5_sybil import windows
from .config import Config
from .metrics import components, cosine_matrix, partial_spearman
from .profiles import miner_profiles
from .subnet import SubnetData


def per_subnet(sd: SubnetData, panel: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    cert = float(panel["cert_share"].mean())
    setters = float(panel["miner_weight_setter_share"].mean())
    commits = float(panel["commitment_share"].mean())
    commit_upd = float(panel["commitment_update_share"].mean())
    axon_upd = float(panel["axon_update_share"].mean()) if panel["has_axon_data"].any() else np.nan
    sig = [cert, setters, commits, commit_upd, axon_upd]
    q1 = {"netuid": sd.netuid, "cert_share": cert, "miner_weight_setter_share": setters,
          "commitment_share": commits, "commitment_update_share": commit_upd,
          "axon_update_share": axon_upd,
          "telemetry_index": float(np.nanmean(sig)),
          # Axon updates are basic liveness (serve_axon), not intent telemetry, so they don't
          # count toward "has miner-level signals"; commitments and certificates do
          "any_onchain_miner_signal": bool((cert > 0.01) or (setters > 0.01) or (commits > 0.01))}

    dups, distinct = [], []
    for a, b in windows(sd, cfg):
        stable = sd.is_miner[a:b].all(axis=0) & ~sd.replaced[a + 1:b].any(axis=0)
        n_m = max(int(stable.sum()), 1)
        share = sd.miner_emi_share[a:b]
        uids = np.where(stable & (share.mean(axis=0) >= cfg.min_miner_share_factor / n_m))[0]
        if len(uids) < 3:
            continue
        traj, wprof = miner_profiles(sd, a, b, uids)
        parts = [traj] + ([wprof] if wprof is not None else [])
        unit = []
        for p_ in parts:
            p_ = p_ - p_.mean(axis=1, keepdims=True)
            n = np.linalg.norm(p_, axis=1, keepdims=True)
            n[n == 0] = 1
            unit.append(p_ / n)
        prof = np.hstack(unit) / np.sqrt(len(unit))
        S = cosine_matrix(prof)
        np.fill_diagonal(S, -1)
        dup = S.max(axis=1) >= cfg.duplicate_similarity
        adj = S >= cfg.duplicate_similarity
        groups = components(adj, 1)
        dups.append(dup.mean())
        distinct.append(len(groups) / len(uids))
    q2 = {"netuid": sd.netuid,
          "indistinguishable_share": float(np.mean(dups)) if dups else np.nan,
          "distinct_profile_ratio": float(np.mean(distinct)) if distinct else np.nan}
    return {"a2_q1_subnets": pd.DataFrame([q1]), "a2_q2_subnets": pd.DataFrame([q2])}


def aggregate(R: Dict[str, pd.DataFrame], panel: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    q1, q2 = R["a2_q1_subnets"], R["a2_q2_subnets"]
    out = {"a2_q1_summary": pd.DataFrame([{
        "subnets": len(q1),
        "share_subnets_with_onchain_miner_signal": q1["any_onchain_miner_signal"].mean(),
        "median_cert_share": q1["cert_share"].median(),
        "median_weight_setter_share": q1["miner_weight_setter_share"].median(),
        "median_commitment_share": q1["commitment_share"].median(),
        "median_axon_update_share": q1["axon_update_share"].median(),
    }]), "a2_q2_summary": pd.DataFrame([{
        "median_indistinguishable_share": q2["indistinguishable_share"].median(),
        "median_distinct_profile_ratio": q2["distinct_profile_ratio"].median(),
    }])}

    # Q3: telemetry vs incident burden
    m = q1.merge(q2, on="netuid", how="outer")
    size = panel.groupby("netuid")["n_uids"].mean().rename("n_uids")
    m = m.merge(size, on="netuid", how="left")
    a1 = R["a1_q1_subnets"][["netuid", "n_incidents"]]
    m = m.merge(a1, on="netuid", how="left")
    inc = R.get("a1_incidents", pd.DataFrame())
    if len(inc):
        g = inc.groupby("netuid").agg(at_risk_subnet_share=("at_risk_subnet_share", "sum"),
                                      median_recovery_hours=("recovery_hours", "median"))
        m = m.merge(g, on="netuid", how="left")
    else:
        m["at_risk_subnet_share"] = 0.0
        m["median_recovery_hours"] = np.nan
    m["at_risk_subnet_share"] = m["at_risk_subnet_share"].fillna(0.0)
    s5 = R["a5_q1_subnets"][["netuid", "cluster_share_mean"]]
    m = m.merge(s5, on="netuid", how="left")
    clu = R.get("a5_clusters", pd.DataFrame())
    if len(clu):
        ex = clu.groupby("netuid").apply(
            lambda g: g["excess_alpha"].sum() / (g["subnet_alpha_flow_window"].sum() or np.nan),
            include_groups=False).rename("sybil_excess_subnet_share")
        m = m.merge(ex, on="netuid", how="left")
    else:
        m["sybil_excess_subnet_share"] = 0.0
    m["sybil_excess_subnet_share"] = m["sybil_excess_subnet_share"].fillna(0.0)
    m["incident_burden"] = m["at_risk_subnet_share"] + m["sybil_excess_subnet_share"]
    out["a2_q3_subnets"] = m

    rows = []
    for x in ("telemetry_index", "indistinguishable_share"):
        for y in ("n_incidents", "median_recovery_hours", "at_risk_subnet_share",
                  "cluster_share_mean", "incident_burden"):
            xv, yv = m[x].to_numpy(float), m[y].to_numpy(float)
            ok = np.isfinite(xv) & np.isfinite(yv)
            if ok.sum() >= 5 and np.ptp(xv[ok]) > 0 and np.ptp(yv[ok]) > 0:
                r = stats.spearmanr(xv[ok], yv[ok])
                rr, rp = float(r.statistic), float(r.pvalue)
            else:
                rr = rp = np.nan
            pr, pp = partial_spearman(xv, yv, m["n_uids"].to_numpy(float))
            rows.append({"telemetry_metric": x, "outcome": y, "n": int(ok.sum()),
                         "spearman_r": rr, "p_value": rp, "partial_r_ctrl_size": pr, "partial_p": pp})
    out["a2_q3_correlations"] = pd.DataFrame(rows)

    # impact estimate: excess burden in the low-telemetry half
    ti = m["telemetry_index"]
    if ti.notna().sum() >= 4 and ti.nunique() > 1:
        low = m[ti <= ti.median()]
        high = m[ti > ti.median()]
    else:  # no variation in telemetry: fall back to distinguishability
        di = m["indistinguishable_share"]
        low = m[di >= di.median()]
        high = m[di < di.median()]
    gap = float(low["incident_burden"].mean() - high["incident_burden"].mean()) if len(low) and len(high) else np.nan
    if len(low) >= 3 and len(high) >= 3:
        p_mw = float(stats.mannwhitneyu(low["incident_burden"], high["incident_burden"],
                                        alternative="greater").pvalue)
    else:
        p_mw = np.nan
    supported = bool(np.isfinite(p_mw) and p_mw < 0.1 and min(len(low), len(high)) >= 5)
    out["a2_q3_summary"] = pd.DataFrame([{
        "low_telemetry_subnets": len(low), "high_telemetry_subnets": len(high),
        "burden_low": low["incident_burden"].mean() if len(low) else np.nan,
        "burden_high": high["incident_burden"].mean() if len(high) else np.nan,
        "excess_burden_low_vs_high": gap,
        "p_value_low_greater": p_mw,
        "evidence_supported": supported,
    }])
    return out
