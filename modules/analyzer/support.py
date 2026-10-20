"""Per-miner validator support, computed once per subnet and cached on the SubnetData.

`Consensus` is the stake-weighted clipped median, and Yuma clips incentive to it, so
emission and Consensus almost always move together. A reward that a few high-stake
validators push through shows up instead as a gap between emission and *breadth* of
support, so these arrays measure support one-validator-one-vote.
"""
from dataclasses import dataclass

import numpy as np

from .subnet import SubnetData, row_share


@dataclass
class Support:
    median_share: np.ndarray   # [T, N] median weight across validators, renormalized over miners
    mean_share: np.ndarray     # [T, N] unweighted mean weight, renormalized over miners
    breadth: np.ndarray        # [T, N] share of validators giving >= a fair share (1 / n_miners)
    cv: np.ndarray             # [T, N] coefficient of variation of received weight
    cons_share: np.ndarray     # [T, N] Consensus storage renormalized over miners
    bonds_share: np.ndarray    # [T, N] bonds received, renormalized over miners


def validator_support(sd: SubnetData) -> Support:
    cached = sd.__dict__.get("_support")
    if cached is not None:
        return cached
    T, N = sd.T, sd.N
    med = np.zeros((T, N))
    mean = np.zeros((T, N))
    breadth = np.zeros((T, N))
    cv = np.zeros((T, N))
    for t in range(T):
        miners = sd.is_miner[t]
        vr = sd.active_validator_rows(t)
        if not len(vr) or not miners.any():
            continue
        Wm = sd.W[t][vr][:, miners]
        Wm = Wm / np.maximum(Wm.sum(axis=1, keepdims=True), 1e-12)
        n_m = miners.sum()
        med[t, miners] = np.median(Wm, axis=0)
        m = Wm.mean(axis=0)
        mean[t, miners] = m
        breadth[t, miners] = (Wm >= 1.0 / n_m).mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cv[t, miners] = np.where(m > 0, Wm.std(axis=0) / m, 0.0)
    cons = np.where(sd.is_miner, np.nan_to_num(sd.consensus), 0.0)
    if sd.bonds.shape[1] > 0:
        bonds = np.where(sd.is_miner, sd.bonds.sum(axis=1), 0.0)
    else:
        bonds = np.zeros((T, N))
    out = Support(row_share(med), row_share(mean), breadth, cv, row_share(cons), row_share(bonds))
    sd.__dict__["_support"] = out
    return out
