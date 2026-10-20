"""Area 5 — Sybil attack vulnerability.

Q1  What share of miners in each subnet belongs to behavioral clusters?
Q2  How much emission do clusters capture relative to their size?
Q3  Are sybil-heavy subnets less stable (and does YC3 change anything)?

Clusters are built per time window from behavior only (trajectory correlation,
received-weight profile, co-registration). Coldkeys / IPs, when present, are used
afterwards to estimate how often a cluster is a single operator.
"""
from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from .adapters import ip_prefix
from .config import Config
from .metrics import components, corr_matrix, partial_spearman
from .profiles import miner_profiles
from .subnet import SubnetData


def _pair_rate(keys: List) -> float:
    keys = [k for k in keys if k]
    if len(keys) < 2:
        return np.nan
    pairs = list(combinations(keys, 2))
    return float(np.mean([a == b for a, b in pairs]))


def windows(sd: SubnetData, cfg: Config):
    W = sd.snaps(cfg.window_days)
    out = []
    for a in range(0, sd.T, W):
        b = min(sd.T, a + W)
        if b - a >= max(10, W // 2):
            out.append((a, b))
    return out


def per_subnet(sd: SubnetData, panel: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    win_rows, clu_rows, mem_rows = [], [], []
    miner_tao = np.where(sd.is_miner, sd.tao_flow, 0).sum(axis=1)
    miner_alpha = np.where(sd.is_miner, sd.alpha_flow, 0).sum(axis=1)

    for wi, (a, b) in enumerate(windows(sd, cfg)):
        stable = sd.is_miner[a:b].all(axis=0) & ~sd.replaced[a + 1:b].any(axis=0)
        n_miners = int(sd.is_miner[a:b].any(axis=0).sum())
        share = sd.miner_emi_share[a:b]                       # [w, N]
        mean_share = share.mean(axis=0)
        cand = stable & (mean_share >= cfg.min_miner_share_factor / max(n_miners, 1)) & (share.std(axis=0) > 0)
        uids = np.where(cand)[0]
        rec = {"netuid": sd.netuid, "window": wi, "t_start": a, "t_end": b,
               "n_miners": n_miners, "n_candidates": len(uids),
               "yc3": float(np.nanmean(sd.yc3[a:b])) if np.isfinite(sd.yc3[a:b]).any() else np.nan}
        clusters = []
        if len(uids) >= cfg.min_cluster_size:
            traj, prof = miner_profiles(sd, a, b, uids)
            C = corr_matrix(traj)
            S = corr_matrix(prof) if prof is not None else np.ones_like(C)
            adj = (C >= cfg.traj_corr) & (S >= cfg.weight_profile_corr)
            if sd.has_reg_block:
                rb = sd.reg_block[a, uids]
                co = np.abs(rb[:, None] - rb[None, :]) <= cfg.coreg_blocks
                adj |= co & (C >= cfg.coreg_traj_corr)
            clusters = components(adj, cfg.min_cluster_size)

        members_all = []
        all_occ = [sd.occupant[a, u] for u in uids]
        base_ck = _pair_rate([sd.coldkey(o) for o in all_occ]) if sd.identity else np.nan
        base_ip = _pair_rate([ip_prefix(sd.ip(o)) for o in all_occ]) if sd.identity else np.nan
        for ci, idx in enumerate(clusters):
            cu = uids[idx]
            members_all.extend(cu.tolist())
            emi = float(mean_share[cu].sum())
            uid_sh = len(cu) / max(n_miners, 1)
            excess = max(emi - uid_sh, 0.0)
            occ = [sd.occupant[a, u] for u in cu]
            c = {"netuid": sd.netuid, "window": wi, "cluster": ci, "size": len(cu),
                 "uid_share": uid_sh, "emission_share": emi,
                 "capture_ratio": emi / uid_sh if uid_sh > 0 else np.nan,
                 "excess_share": excess,
                 "excess_tao": excess * miner_tao[a:b].sum(),
                 "excess_alpha": excess * miner_alpha[a:b].sum(),
                 "subnet_alpha_flow_window": float(sd.alpha_flow[a:b].sum()),
                 "yc3": rec["yc3"]}
            if sd.has_reg_block:
                rb = sd.reg_block[a, cu]
                c["reg_span_blocks"] = float(np.nanmax(rb) - np.nanmin(rb))
                med = np.nanmedian(rb)
                c["onset_t"] = int(np.searchsorted(sd.blocks, med)) if sd.blocks[0] <= med <= sd.blocks[-1] else np.nan
            if sd.identity:
                c["same_coldkey_pair_rate"] = _pair_rate([sd.coldkey(o) for o in occ])
                c["same_ip24_pair_rate"] = _pair_rate([ip_prefix(sd.ip(o)) for o in occ])
                c["baseline_coldkey_pair_rate"] = base_ck
                c["baseline_ip24_pair_rate"] = base_ip
            clu_rows.append(c)
            for u, o in zip(cu, occ):
                mem_rows.append({"netuid": sd.netuid, "window": wi, "cluster": ci, "uid": int(u), "occupant": o})

        in_cluster = np.zeros(sd.N, bool)
        in_cluster[members_all] = True
        rec["n_clustered"] = int(in_cluster.sum())
        rec["cluster_share"] = in_cluster.sum() / max(n_miners, 1)
        rec["n_clusters"] = len(clusters)
        rec["largest_cluster"] = max((len(i) for i in clusters), default=0)
        non = sd.is_miner[a:b] & ~in_cluster[None, :]
        rec["non_cluster_turnover"] = float(sd.replaced[a:b][non].mean()) if non.any() else np.nan
        win_rows.append(rec)

    wdf = pd.DataFrame(win_rows)
    summary = pd.DataFrame([{
        "netuid": sd.netuid,
        "cluster_share_mean": wdf["cluster_share"].mean() if len(wdf) else np.nan,
        "cluster_share_max": wdf["cluster_share"].max() if len(wdf) else np.nan,
        "n_clusters_mean": wdf["n_clusters"].mean() if len(wdf) else np.nan,
        "non_cluster_turnover": wdf["non_cluster_turnover"].mean() if len(wdf) else np.nan,
        "yc3_on_share": wdf["yc3"].mean() if len(wdf) else np.nan,
    }])
    return {"a5_windows": wdf, "a5_clusters": pd.DataFrame(clu_rows),
            "a5_members": pd.DataFrame(mem_rows), "a5_q1_subnets": summary}


def aggregate(R: Dict[str, pd.DataFrame], panel: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    s = R["a5_q1_subnets"].copy()
    clu = R.get("a5_clusters", pd.DataFrame())
    out = {}

    q1 = {"subnets": len(s),
          "share_subnets_any_cluster": float((s["cluster_share_max"] > 0).mean()),
          "share_subnets_above_threshold": float((s["cluster_share_max"] >= cfg.sybil_share_threshold).mean()),
          "median_cluster_share": float(s["cluster_share_mean"].median()),
          "p90_cluster_share": float(s["cluster_share_max"].quantile(0.9))}
    if len(clu) and "same_coldkey_pair_rate" in clu:
        q1["cluster_same_coldkey_rate"] = float(clu["same_coldkey_pair_rate"].mean())
        q1["baseline_same_coldkey_rate"] = float(clu["baseline_coldkey_pair_rate"].mean())
        q1["cluster_same_ip24_rate"] = float(clu["same_ip24_pair_rate"].mean())
        q1["baseline_same_ip24_rate"] = float(clu["baseline_ip24_pair_rate"].mean())
    out["a5_q1_summary"] = pd.DataFrame([q1])

    if len(clu):
        by = clu.groupby("netuid").agg(clusters=("cluster", "count"),
                                       median_capture_ratio=("capture_ratio", "median"),
                                       excess_tao=("excess_tao", "sum"),
                                       excess_alpha=("excess_alpha", "sum")).reset_index()
        out["a5_q2_by_subnet"] = by
        out["a5_q2_summary"] = pd.DataFrame([{
            "clusters": len(clu),
            "median_capture_ratio": float(clu["capture_ratio"].median()),
            "share_clusters_ratio_above_1_5": float((clu["capture_ratio"] > 1.5).mean()),
            "total_excess_tao": float(clu["excess_tao"].sum()),
        }])

    # Q3: cross-subnet stability correlations
    stab = panel.groupby("netuid").agg(rank_corr=("rank_corr", "mean"),
                                       agreement=("validator_agreement", "mean"),
                                       n_uids=("n_uids", "mean")).reset_index()
    m = s.merge(stab, on="netuid")
    m["rank_volatility"] = 1 - m["rank_corr"]
    m["validator_disagreement"] = 1 - m["agreement"]
    rows = []
    for y in ("rank_volatility", "validator_disagreement", "non_cluster_turnover"):
        x, yy = m["cluster_share_mean"].to_numpy(float), m[y].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(yy)
        r = stats.spearmanr(x[ok], yy[ok]) if ok.sum() >= 5 and np.ptp(x[ok]) > 0 and np.ptp(yy[ok]) > 0 else None
        pr, pp = partial_spearman(x, yy, m["n_uids"].to_numpy(float))
        rows.append({"outcome": y, "n": int(ok.sum()),
                     "spearman_r": float(r.statistic) if r is not None else np.nan,
                     "p_value": float(r.pvalue) if r is not None else np.nan,
                     "partial_r_ctrl_size": pr, "partial_p": pp})
    out["a5_q3_stability"] = pd.DataFrame(rows)

    # YC3 comparison (needs Yuma3On collected)
    yc = []
    if len(clu) and clu["yc3"].notna().any():
        on = clu.loc[clu["yc3"] >= 0.5, "capture_ratio"].dropna()
        off = clu.loc[clu["yc3"] < 0.5, "capture_ratio"].dropna()
        if len(on) >= 3 and len(off) >= 3:
            u = stats.mannwhitneyu(on, off)
            yc.append({"comparison": "clusters YC3 on vs off", "n_on": len(on), "n_off": len(off),
                       "median_ratio_on": on.median(), "median_ratio_off": off.median(), "p_value": u.pvalue})
        w = R["a5_windows"]
        for net, g in w.groupby("netuid"):
            if g["yc3"].notna().any() and g["yc3"].nunique() > 1:
                yc.append({"comparison": f"netuid {net} toggled YC3",
                           "share_before": g.loc[g["yc3"] < 0.5, "cluster_share"].mean(),
                           "share_after": g.loc[g["yc3"] >= 0.5, "cluster_share"].mean()})
    out["a5_q3_yc3"] = pd.DataFrame(yc)
    return out
