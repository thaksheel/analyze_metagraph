"""Miner-validator-subnet (MVS) interactions: do validators agree, score independently, and
does disagreement hurt miners?"""
from typing import Dict

import numpy as np
import pandas as pd

from .base import Analysis


class ScoringAnalysis(Analysis):
    key = "scoring"
    title = "Understanding MVS interactions"
    problem = "understanding mvs interactions"
    hypothesis = "mis-scoring leads to poor miner selection"
    questions = {
        "Q1": "How much do validators disagree?",
        "Q2": "How much scoring is real, independent judgment?",
        "Q3": "Does disagreement lead to bad miner outcomes?",
    }
    tables = {
        "subnets": ("Q1-Q3", "per subnet: agreement, contested miners, stale/copy-like validators, outcomes"),
        "summary": ("Q1-Q3", "medians across subnets"),
    }

    def per_subnet(self, sn, panel, ctx):
        c, g = self.cfg.scoring, self.cfg.general
        fl = ctx["flags"]
        p = panel
        rec = {
            "netuid": sn.netuid,
            # Q1
            "mean_validators": p["n_validators"].mean(),
            "validator_agreement": p["validator_agreement"].mean(),
            "validator_rank_agreement": p["validator_rank_agreement"].mean(),
            "contested_share": p["contested_share"].mean(),
            "low_agreement": bool(p["validator_agreement"].mean() < c.low_agreement),
            "stake_concentration": p["stake_concentration"].mean(),
            # Q2
            "stale_validator_share": p["stale_validator_share"].mean(),
            "copy_validator_share": p["copy_validator_share"].mean(),
            "stale_div_share": p["stale_div_share"].mean(),
            "copy_div_share": p["copy_div_share"].mean(),
            "non_independent_div_share": (p["stale_div_share"] + p["copy_div_share"]).clip(upper=1).mean(),
            "effective_scorers": p["effective_scorers"].mean(),
            "effective_scorer_ratio": (p["effective_scorers"] / p["n_validators"].replace(0, np.nan)).mean(),
            "commit_reveal_on": bool(np.nanmax(sn.commit_reveal) > 0) if np.isfinite(sn.commit_reveal).any() else False,
        }
        # Q3: emission to contested miners, and are they replaced more often?
        m_tao = np.where(sn.is_miner, sn.tao_flow, 0.0)
        m_alpha = np.where(sn.is_miner, sn.alpha_flow, 0.0)
        rec["contested_emission_share"] = p["contested_emission_share"].mean()
        rec["contested_tao"] = float((m_tao * fl.contested).sum())
        rec["contested_alpha"] = float((m_alpha * fl.contested).sum())
        rec["contested_subnet_share"] = rec["contested_alpha"] / (sn.alpha_flow.sum() or np.nan)
        H = sn.snaps(c.outcome_days)
        cons = np.nan_to_num(sn.consensus)
        parts = []
        for t in range(1, sn.T - H, max(1, H // 2)):
            miners = sn.is_miner[t]
            relevant = miners & (fl.received_mean[t] >= g.min_miner_share_factor / max(int(miners.sum()), 1))
            if relevant.sum() < 5:
                continue
            uids = np.where(relevant)[0]
            q = pd.qcut(pd.Series(cons[t, uids]).rank(method="first"), 5, labels=False)
            parts.append(pd.DataFrame({"bin": q.to_numpy(), "contested": fl.contested[t, uids],
                                       "replaced": sn.replaced[t + 1:t + H + 1][:, uids].any(axis=0)}))
        rec["replace_rate_contested"] = rec["replace_rate_uncontested"] = rec["replace_rate_gap"] = np.nan
        if parts:
            gtab = pd.concat(parts).groupby(["bin", "contested"])["replaced"].mean().unstack().dropna()
            if len(gtab) and True in gtab and False in gtab:
                rec["replace_rate_contested"] = float(gtab[True].mean())
                rec["replace_rate_uncontested"] = float(gtab[False].mean())
                rec["replace_rate_gap"] = rec["replace_rate_contested"] - rec["replace_rate_uncontested"]
        return {"subnets": pd.DataFrame([rec])}

    def aggregate(self, results, panel):
        s = self.combined("subnets")
        return {"summary": pd.DataFrame([{
            "median_validator_agreement": s["validator_agreement"].median(),
            "share_subnets_low_agreement": s["low_agreement"].mean(),
            "median_contested_share": s["contested_share"].median(),
            "median_stake_concentration": s["stake_concentration"].median(),
            "median_non_independent_div_share": s["non_independent_div_share"].median(),
            "median_effective_scorer_ratio": s["effective_scorer_ratio"].median(),
            "share_subnets_commit_reveal": s["commit_reveal_on"].mean(),
            "median_contested_emission_share": s["contested_emission_share"].median(),
            "total_contested_tao": s["contested_tao"].sum(),
            "median_replace_rate_gap": s["replace_rate_gap"].median(),
            "share_subnets_contested_replaced_more": (s["replace_rate_gap"] > 0).mean(),
        }])}

    def rank_entry(self, results) -> Dict:
        """Size: share of subnets with low agreement. Impact: contested emission x disagreement."""
        s = results.get("scoring_subnets").set_index("netuid")
        dis = (1 - s["validator_agreement"]).clip(lower=0).fillna(0)
        summ = results.get("scoring_summary").iloc[0]
        return {"size_metric": "share of subnets with low validator agreement",
                "size_value": float(s["low_agreement"].mean()),
                "affected_alpha": s["contested_alpha"] * dis, "affected_tao": s["contested_tao"] * dis,
                "upper_alpha": s["contested_alpha"], "upper_tao": s["contested_tao"],
                "measured": f"median replace-rate gap {summ['median_replace_rate_gap']:.3f}; "
                            f"{summ['share_subnets_commit_reveal']:.0%} of subnets use commit-reveal"}
