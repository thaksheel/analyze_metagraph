"""Panel: one row per (subnet, snapshot) of derived metrics, shared by every analysis.

    panel, flags = PanelBuilder(cfg).build(subnet)

`flags` holds the per-uid arrays later analyses reuse (contested miners, received
weight, stale / copy-like validators). SIGNALS lists the panel columns treated as
"metagraph signals" by the insight analysis.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..data.subnet import Subnet
from .stats import cosine_matrix, gini, hhi, spearman, top_k_share

SIGNALS = [
    "n_uids", "n_miners", "gini_inc", "top10_share", "hhi_emission", "active_share", "rank_corr",
    "replacement_rate", "burn_share", "n_validators", "burn_weight_mass", "validator_agreement",
    "validator_rank_agreement", "contested_share", "contested_emission_share", "stale_validator_share",
    "copy_validator_share", "stale_div_share", "copy_div_share", "effective_scorers",
    "div_participation", "stake_concentration", "vtrust_mean", "cons_inc_gap",
    "miner_weight_setter_share", "price", "subnet_tao_flow", "kappa", "rho", "tempo",
    "activity_cutoff", "burn_cost",
]
HYPERPARAMS = ["tempo", "kappa", "rho", "activity_cutoff", "mechanism", "mechanism_count",
               "immunity_period", "max_validators", "yc3", "commit_reveal"]


@dataclass
class UidFlags:
    contested: np.ndarray      # [T, N] miner's received weight varies widely across validators
    received_mean: np.ndarray  # [T, N] mean weight received from active validators
    stale: np.ndarray          # [T, N] validator stale at t
    copy_like: np.ndarray      # [T, N] validator persistently tracks the previous consensus


def _weighted_pair_mean(S: np.ndarray, w: np.ndarray) -> float:
    k = len(w)
    if k < 2:
        return np.nan
    Wm = np.outer(w, w)
    np.fill_diagonal(Wm, 0)
    if Wm.sum() <= 0:
        return float(S[np.triu_indices(k, 1)].mean())
    return float((S * Wm).sum() / Wm.sum())


class PanelBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build(self, sn: Subnet) -> Tuple[pd.DataFrame, UidFlags]:
        sc, g = self.cfg.scoring, self.cfg.general
        T, N = sn.T, sn.N
        rows = sn.weights_rows
        contested = np.zeros((T, N), bool)
        received = np.zeros((T, N))
        stale = np.zeros((T, N), bool)
        lag_closer = np.zeros((T, N), bool)
        vstate, recs = [], []
        cons = np.nan_to_num(sn.consensus)
        last_up = sn.last_update

        for t in range(T):
            miners = sn.is_miner[t]
            n_m = max(int(miners.sum()), 1)
            inc_m = np.where(miners, np.nan_to_num(sn.incentive[t]), 0.0)
            r = {"netuid": sn.netuid, "t": t, "block": int(sn.blocks[t]), "timestamp": sn.timestamps[t],
                 "hours": sn.hours[t], "n_uids": int(sn.exists[t].sum()), "n_miners": int(miners.sum())}
            r["gini_inc"] = gini(inc_m[miners]) if miners.any() else np.nan
            r["top10_share"] = top_k_share(sn.emission[t][sn.exists[t]])
            r["hhi_emission"] = hhi(sn.emission[t][sn.exists[t]])
            r["active_share"] = float(np.nanmean(sn.active[t][sn.exists[t]])) if sn.exists[t].any() else np.nan
            r["rank_corr"] = np.nan
            if t > 0:
                same = miners & sn.is_miner[t - 1] & ~sn.replaced[t]
                r["rank_corr"] = spearman(sn.incentive[t - 1][same], sn.incentive[t][same])
            r["replacement_rate"] = float(sn.replaced[t][miners].mean()) if miners.any() else np.nan
            ou = sn.owner_uid[t]
            r["burn_share"] = float(sn.inc_share[t, ou]) if N else np.nan

            vr = sn.active_validator_rows(t)
            vuids = rows[vr] if len(vr) else np.array([], int)
            r["n_validators"] = len(vuids)
            nan_keys = ("burn_weight_mass", "validator_agreement", "validator_rank_agreement",
                        "contested_share", "contested_emission_share", "stale_validator_share",
                        "stale_div_share", "div_participation", "stake_concentration", "vtrust_mean")
            if len(vuids):
                Wt = sn.W[t][vr]
                w_v = sn.validator_weight(t, vuids)
                r["burn_weight_mass"] = float(Wt[:, ou].mean())
                r["validator_agreement"] = _weighted_pair_mean(cosine_matrix(Wt), w_v)
                r["validator_rank_agreement"] = (_weighted_pair_mean(np.corrcoef(stats.rankdata(Wt, axis=1)), w_v)
                                                 if len(vuids) >= 2 else np.nan)
                col = Wt[:, miners]
                mean, sd_ = col.mean(axis=0), col.std(axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cv = np.where(mean > 0, sd_ / mean, 0.0)
                cmask = np.zeros(N, bool)
                cmask[np.where(miners)[0]] = (mean >= g.min_miner_share_factor / n_m) & (cv > sc.contested_cv) & (len(vuids) >= 2)
                contested[t] = cmask
                received[t, miners] = mean
                r["contested_share"] = float(cmask[miners].mean())
                r["contested_emission_share"] = float(sn.miner_emi_share[t][cmask].sum())
                lag = sn.blocks[t] - last_up[t, vuids]
                st = np.nan_to_num(lag, nan=np.inf) > sc.stale_tempos * sn.tempo_v[t]
                cprev = cons[t - 1] if t > 0 else cons[t]
                nW = np.linalg.norm(Wt, axis=1)
                cos_prev = (Wt @ cprev) / (nW * (np.linalg.norm(cprev) or 1) + 1e-12)
                cos_now = (Wt @ cons[t]) / (nW * (np.linalg.norm(cons[t]) or 1) + 1e-12)
                lag_closer[t, vuids] = (cos_prev >= sc.copy_similarity) & (cos_prev > cos_now) & (t > 0)
                stale[t, vuids] = st
                dv = sn.div_share[t, vuids]
                dtot = dv.sum() or 1.0
                vstate.append((t, vuids, st, dv / dtot))
                r["stale_validator_share"] = float(st.mean())
                r["stale_div_share"] = float(dv[st].sum() / dtot)
                r["div_participation"] = float(1.0 / (((dv / dtot) ** 2).sum() or 1.0))
                ws = w_v / (w_v.sum() or 1)
                r["stake_concentration"] = float((ws ** 2).sum())   # HHI of validator influence
                r["vtrust_mean"] = float(np.nanmean(sn.vtrust[t, vuids]))
            else:
                for k in nan_keys:
                    r[k] = np.nan
            cs = cons[t] / (cons[t].sum() or 1)
            r["cons_inc_gap"] = float(np.abs(cs - sn.inc_share[t]).sum() / 2)
            r["miner_weight_setter_share"] = np.nan
            if t > 0:
                moved = np.nan_to_num(last_up[t]) > np.nan_to_num(last_up[t - 1])
                r["miner_weight_setter_share"] = float((moved & miners & ~sn.replaced[t]).sum() / n_m)
            r["price"] = float(sn.price[t])
            r["subnet_tao_flow"] = float(sn.subnet_tao_flow[t])
            r["subnet_alpha_flow"] = float(sn.alpha_flow[t].sum())
            r["miner_tao_flow"] = float(sn.tao_flow[t][miners].sum())
            r["miner_alpha_flow"] = float(sn.alpha_flow[t][miners].sum())
            for hp in HYPERPARAMS + ["burn_cost"]:
                r[hp] = float(getattr(sn, hp)[t])
            recs.append(r)

        # copy-like = persistently closer to the previous consensus than the current one
        K = sn.snaps(2.0)
        copy_like = np.zeros((T, N), bool)
        csum = np.cumsum(np.vstack([np.zeros((1, N)), lag_closer.astype(float)]), axis=0)
        for t, vuids, st, dvn in vstate:
            lo = max(0, t - K + 1)
            frac = (csum[t + 1, vuids] - csum[lo, vuids]) / (t + 1 - lo)
            cp = lag_closer[t, vuids] & (frac >= sc.copy_persistence) & (t + 1 - lo >= min(K, 4))
            copy_like[t, vuids] = cp
            recs[t]["copy_validator_share"] = float(cp.mean())
            recs[t]["copy_div_share"] = float(dvn[cp].sum())
            recs[t]["effective_scorers"] = int((~st & ~cp).sum())
        for rec in recs:
            for k in ("copy_validator_share", "copy_div_share", "effective_scorers"):
                rec.setdefault(k, np.nan)
        return pd.DataFrame(recs), UidFlags(contested, received, stale, copy_like)
