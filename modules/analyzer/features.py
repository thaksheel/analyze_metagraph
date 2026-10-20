"""Per-snapshot, per-subnet metrics shared by all areas.

`compute_subnet` returns one row per snapshot (the "panel") plus per-UID arrays
(contested miners, copy-like / stale validators) that later questions reuse.
"""
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config
from .metrics import cosine_matrix, gini, hhi, spearman, top_k_share
from .subnet import SubnetData


@dataclass
class SubnetExtras:
    contested: np.ndarray      # [T, N] miner contested at t
    received_mean: np.ndarray  # [T, N] mean weight received from active validators
    stale: np.ndarray          # [T, N] validator stale at t
    copy_like: np.ndarray      # [T, N] validator weights track lagged consensus
    rank_corr: np.ndarray      # [T]


def _weighted_pair_mean(S: np.ndarray, w: np.ndarray) -> float:
    k = len(w)
    if k < 2:
        return np.nan
    Wm = np.outer(w, w)
    np.fill_diagonal(Wm, 0)
    tot = Wm.sum()
    if tot <= 0:
        iu = np.triu_indices(k, 1)
        return float(S[iu].mean())
    return float((S * Wm).sum() / tot)


def compute_subnet(sd: SubnetData, cfg: Config):
    T, N = sd.T, sd.N
    rows = sd.weights_rows
    contested = np.zeros((T, N), bool)
    received_mean = np.zeros((T, N))
    stale = np.zeros((T, N), bool)
    copy_like = np.zeros((T, N), bool)
    rank_corr = np.full(T, np.nan)
    lag_closer = np.zeros((T, N), bool)
    vstate = []
    recs = []

    cons_share = np.nan_to_num(sd.consensus)
    vtrust = np.nan_to_num(sd.vtrust)
    last_up = sd.last_update

    for t in range(T):
        miners = sd.is_miner[t]
        n_miners = max(int(miners.sum()), 1)
        inc_m = np.where(miners, np.nan_to_num(sd.incentive[t]), 0.0)
        r = {"netuid": sd.netuid, "t": t, "block": int(sd.blocks[t]),
             "timestamp": sd.timestamps[t], "hours": sd.hours[t]}
        r["n_uids"] = int(sd.exists[t].sum())
        r["n_miners"] = int(miners.sum())
        r["gini_inc"] = gini(inc_m[miners]) if miners.any() else np.nan
        r["top10_share"] = top_k_share(sd.emission[t][sd.exists[t]])
        r["hhi_emission"] = hhi(sd.emission[t][sd.exists[t]])
        r["active_share"] = float(np.nanmean(sd.active[t][sd.exists[t]])) if sd.exists[t].any() else np.nan

        # rank stability among miners whose occupant did not change
        if t > 0:
            same = miners & sd.is_miner[t - 1] & ~sd.replaced[t]
            rank_corr[t] = spearman(sd.incentive[t - 1][same], sd.incentive[t][same])
        r["rank_corr"] = rank_corr[t]
        r["replacement_rate"] = float(sd.replaced[t][miners].mean()) if miners.any() else np.nan

        # burn: incentive routed to the owner / burn UID
        ou = sd.owner_uid[t]
        r["burn_share"] = float(sd.inc_share[t, ou]) if N else np.nan

        # validators
        vr = sd.active_validator_rows(t)
        vuids = rows[vr] if len(vr) else np.array([], int)
        r["n_validators"] = len(vuids)
        if len(vuids):
            Wt = sd.W[t][vr]                          # [V, N]
            w_v = vtrust[t, vuids]
            r["burn_weight_mass"] = float(Wt[:, ou].mean())
            S = cosine_matrix(Wt)
            r["validator_agreement"] = _weighted_pair_mean(S, w_v)
            if len(vuids) >= 2:
                R = stats.rankdata(Wt, axis=1)
                r["validator_rank_agreement"] = _weighted_pair_mean(np.corrcoef(R), w_v)
            else:
                r["validator_rank_agreement"] = np.nan

            # contested miners: high dispersion of received weight across validators
            col = Wt[:, miners]
            mean = col.mean(axis=0)
            sd_ = col.std(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                cv = np.where(mean > 0, sd_ / mean, 0.0)
            relevant = mean >= cfg.min_miner_share_factor / n_miners
            cmask = np.zeros(N, bool)
            cmask[np.where(miners)[0]] = relevant & (cv > cfg.contested_cv) & (len(vuids) >= 2)
            contested[t] = cmask
            received_mean[t, miners] = mean
            r["contested_share"] = float(cmask[miners].mean())
            r["contested_emission_share"] = float(sd.miner_emi_share[t][cmask].sum())

            # stale and copy-like validators
            lag = sd.blocks[t] - last_up[t, vuids]
            st = np.nan_to_num(lag, nan=np.inf) > cfg.stale_tempos * sd.tempo_v[t]
            cons_prev = cons_share[t - 1] if t > 0 else cons_share[t]
            cos_prev = (Wt @ cons_prev) / (np.linalg.norm(Wt, axis=1) * (np.linalg.norm(cons_prev) or 1) + 1e-12)
            cos_now = (Wt @ cons_share[t]) / (np.linalg.norm(Wt, axis=1) * (np.linalg.norm(cons_share[t]) or 1) + 1e-12)
            lag_closer[t, vuids] = (cos_prev >= cfg.copy_similarity) & (cos_prev > cos_now) & (t > 0)
            stale[t, vuids] = st
            dv = sd.div_share[t, vuids]
            dtot = dv.sum() or 1.0
            vstate.append((t, vuids, st, dv / dtot))
            r["stale_validator_share"] = float(st.mean())
            r["stale_div_share"] = float(dv[st].sum() / dtot)
            r["div_participation"] = float(1.0 / (((dv / dtot) ** 2).sum() or 1.0))
            r["vtrust_mean"] = float(np.nanmean(sd.vtrust[t, vuids]))
        else:
            for k in ("burn_weight_mass", "validator_agreement", "validator_rank_agreement",
                      "contested_share", "contested_emission_share", "stale_validator_share",
                      "copy_validator_share", "stale_div_share", "copy_div_share",
                      "effective_scorers", "div_participation", "vtrust_mean"):
                r[k] = np.nan

        # consensus vs incentive gap
        cs = cons_share[t] / (cons_share[t].sum() or 1)
        r["cons_inc_gap"] = float(np.abs(cs - sd.inc_share[t]).sum() / 2)

        # miner-emitted signals
        if sd.cert_flag.any():
            r["cert_share"] = float(sd.cert_flag[t][miners].mean()) if miners.any() else np.nan
        else:
            r["cert_share"] = min(1.0, float(sd.cert_count[t]) / n_miners)
        if t > 0:
            moved = np.nan_to_num(last_up[t]) > np.nan_to_num(last_up[t - 1])
            r["miner_weight_setter_share"] = float((moved & miners & ~sd.replaced[t]).sum() / n_miners)
        else:
            r["miner_weight_setter_share"] = np.nan

        # other miner-emitted on-chain signals (collected with the updated collector)
        ab, cb = sd.axon_block, sd.commit_block
        r["commitment_share"] = float(np.isfinite(cb[t][miners]).mean()) if miners.any() else np.nan
        if t > 0 and miners.any():
            same = miners & ~sd.replaced[t]
            with np.errstate(invalid="ignore"):
                r["axon_update_share"] = float((same & (np.nan_to_num(ab[t]) > np.nan_to_num(ab[t - 1]))).sum() / n_miners)
                r["commitment_update_share"] = float((same & (np.nan_to_num(cb[t]) > np.nan_to_num(cb[t - 1]))).sum() / n_miners)
        else:
            r["axon_update_share"] = r["commitment_update_share"] = np.nan
        r["has_axon_data"] = bool(np.isfinite(ab[t]).any())
        r["price"] = float(sd.price[t])
        r["subnet_tao_flow"] = float(sd.subnet_tao_flow[t])
        r["subnet_alpha_flow"] = float(sd.alpha_flow[t].sum())
        r["miner_tao_flow"] = float(sd.tao_flow[t][miners].sum())
        r["miner_alpha_flow"] = float(sd.alpha_flow[t][miners].sum())
        for hp in ("kappa", "rho", "tempo", "activity_cutoff", "mechanism", "mechanism_count", "yc3"):
            r[hp] = float(getattr(sd, hp)[t])
        recs.append(r)

    # copy-like = validator persistently closer to the *previous* consensus than the
    # current one (honest validators pull consensus toward themselves, so this is rare)
    K = sd.snaps(2.0)
    csum = np.cumsum(np.vstack([np.zeros((1, N)), lag_closer.astype(float)]), axis=0)
    for t, vuids, st, dvn in vstate:
        lo = max(0, t - K + 1)
        frac = (csum[t + 1, vuids] - csum[lo, vuids]) / (t + 1 - lo)
        cp = lag_closer[t, vuids] & (frac >= 0.8) & (t + 1 - lo >= min(K, 4))
        copy_like[t, vuids] = cp
        rec = recs[t]
        rec["copy_validator_share"] = float(cp.mean())
        rec["copy_div_share"] = float(dvn[cp].sum())
        rec["effective_scorers"] = int((~st & ~cp).sum())
    for rec in recs:
        for k in ("copy_validator_share", "copy_div_share", "effective_scorers"):
            rec.setdefault(k, np.nan)
    panel = pd.DataFrame(recs)
    return panel, SubnetExtras(contested, received_mean, stale, copy_like, rank_corr)


# derived signals used in area 4 (subnet-level, per snapshot)
SIGNAL_COLUMNS = [
    "n_uids", "n_miners", "gini_inc", "top10_share", "hhi_emission", "active_share",
    "rank_corr", "replacement_rate", "burn_share", "n_validators", "burn_weight_mass",
    "validator_agreement", "validator_rank_agreement", "contested_share",
    "contested_emission_share", "stale_validator_share", "copy_validator_share",
    "stale_div_share", "copy_div_share", "effective_scorers", "div_participation",
    "vtrust_mean", "cons_inc_gap", "cert_share", "miner_weight_setter_share",
    "commitment_share", "axon_update_share", "commitment_update_share", "price",
    "subnet_tao_flow", "kappa", "rho", "tempo", "activity_cutoff",
]
