"""Area 3 — Understanding MVS interactions.

Q1  How much do validators disagree?
Q2  How much scoring is real, independent judgment?
Q3  Does disagreement lead to bad miner outcomes?
"""
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config
from .features import SubnetExtras
from .subnet import SubnetData

OUTCOME_DAYS = 7.0


def per_subnet(sd: SubnetData, panel: pd.DataFrame, ex: SubnetExtras, cfg: Config) -> Dict[str, pd.DataFrame]:
    p = panel
    q1 = {
        "netuid": sd.netuid,
        "mean_validators": p["n_validators"].mean(),
        "validator_agreement": p["validator_agreement"].mean(),
        "validator_rank_agreement": p["validator_rank_agreement"].mean(),
        "contested_share": p["contested_share"].mean(),
        "low_agreement": bool(p["validator_agreement"].mean() < cfg.low_agreement),
    }
    q2 = {
        "netuid": sd.netuid,
        "stale_validator_share": p["stale_validator_share"].mean(),
        "copy_validator_share": p["copy_validator_share"].mean(),
        "stale_div_share": p["stale_div_share"].mean(),
        "copy_div_share": p["copy_div_share"].mean(),
        "non_independent_div_share": (p["stale_div_share"] + p["copy_div_share"]).clip(upper=1).mean(),
        "effective_scorers": p["effective_scorers"].mean(),
        "effective_scorer_ratio": (p["effective_scorers"] / p["n_validators"].replace(0, np.nan)).mean(),
        "div_participation": p["div_participation"].mean(),
    }

    # Q3: emission to contested miners and replacement outcomes
    miner_tao = np.where(sd.is_miner, sd.tao_flow, 0.0)
    miner_alpha = np.where(sd.is_miner, sd.alpha_flow, 0.0)
    contested_tao = float((miner_tao * ex.contested).sum())
    contested_alpha = float((miner_alpha * ex.contested).sum())
    H = sd.snaps(OUTCOME_DAYS)
    step = max(1, H // 2)
    rows = []
    cons = np.nan_to_num(sd.consensus)
    for t in range(1, sd.T - H, step):
        miners = sd.is_miner[t]
        n_m = max(int(miners.sum()), 1)
        relevant = miners & (ex.received_mean[t] >= cfg.min_miner_share_factor / n_m)
        if relevant.sum() < 5:
            continue
        uids = np.where(relevant)[0]
        out = sd.replaced[t + 1:t + H + 1][:, uids].any(axis=0)
        q = pd.qcut(pd.Series(cons[t, uids]).rank(method="first"), 5, labels=False)
        rows.append(pd.DataFrame({"bin": q.to_numpy(), "contested": ex.contested[t, uids], "replaced": out}))
    rate_c = rate_u = gap = np.nan
    if rows:
        d = pd.concat(rows)
        g = d.groupby(["bin", "contested"])["replaced"].mean().unstack()
        if True in g.columns and False in g.columns:
            both = g.dropna()
            if len(both):
                rate_c = float(both[True].mean())
                rate_u = float(both[False].mean())
                gap = rate_c - rate_u
    q3 = {
        "netuid": sd.netuid,
        "contested_emission_share": p["contested_emission_share"].mean(),
        "contested_tao": contested_tao,
        "contested_alpha": contested_alpha,
        "contested_subnet_share": contested_alpha / (sd.alpha_flow.sum() or np.nan),
        "replace_rate_contested": rate_c,
        "replace_rate_uncontested": rate_u,
        "replace_rate_gap": gap,
    }
    return {"a3_q1_subnets": pd.DataFrame([q1]), "a3_q2_subnets": pd.DataFrame([q2]),
            "a3_q3_subnets": pd.DataFrame([q3])}


def aggregate(R: Dict[str, pd.DataFrame], cfg: Config) -> Dict[str, pd.DataFrame]:
    q1, q2, q3 = R["a3_q1_subnets"], R["a3_q2_subnets"], R["a3_q3_subnets"]
    return {"a3_summary": pd.DataFrame([{
        "median_validator_agreement": q1["validator_agreement"].median(),
        "share_subnets_low_agreement": q1["low_agreement"].mean(),
        "median_contested_share": q1["contested_share"].median(),
        "median_non_independent_div_share": q2["non_independent_div_share"].median(),
        "median_effective_scorer_ratio": q2["effective_scorer_ratio"].median(),
        "median_contested_emission_share": q3["contested_emission_share"].median(),
        "total_contested_tao": q3["contested_tao"].sum(),
        "median_replace_rate_gap": q3["replace_rate_gap"].median(),
        "share_subnets_contested_replaced_more": (q3["replace_rate_gap"] > 0).mean(),
    }])}
