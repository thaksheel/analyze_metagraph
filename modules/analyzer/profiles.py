"""Behavioral miner profiles shared by Area 2 (distinguishability) and Area 5 (clusters)."""
from typing import Optional, Tuple

import numpy as np

from .subnet import SubnetData

EPS = 1e-6


def miner_profiles(sd: SubnetData, a: int, b: int, uids: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (trajectory [n, w], validator-preference profile [n, K] or None).

    trajectory: log emission share minus the cross-sectional median at each snapshot,
        which removes common-mode moves (e.g. everyone's share shrinking when one
        cohort grabs emission) that would otherwise correlate unrelated miners.
    profile: how much each validator favors the miner relative to the other
        validators (log weight centered across validators), averaged over the window.
    """
    share = sd.miner_emi_share[a:b]
    ls = np.log(share + EPS)
    miners = sd.is_miner[a:b]
    med = np.array([np.median(ls[t][miners[t]]) if miners[t].any() else 0.0 for t in range(b - a)])
    traj = (ls[:, uids] - med[:, None]).T
    prof = None
    if len(sd.weights_rows):
        Wv = sd.W[a:b][:, :, uids]                       # [w, K, n]
        used = Wv.sum(axis=(0, 2)) > 0
        if used.sum() >= 2:
            lw = np.log(Wv[:, used, :] + EPS)
            rel = lw - lw.mean(axis=1, keepdims=True)
            prof = rel.mean(axis=0).T                    # [n, K]
    return traj, prof
