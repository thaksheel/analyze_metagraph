import numpy as np
import pandas as pd

from ..data.subnet import row_share


def working_mask(sn, min_share=0.001):
    """Working miner: not a validator, not the owner uid, and >= min_share of miner rewards."""
    reg = sn.exists & ~sn.is_validator
    share = row_share(np.where(reg, np.nan_to_num(sn.incentive), 0.0))
    m = reg & (share >= min_share)
    if sn.has_hotkeys and (sn.owner_hotkey != "").any():
        rows = np.arange(sn.T)
        own = sn.hotkeys[rows, sn.owner_uid] == sn.owner_hotkey
        m[rows[own], sn.owner_uid[own]] = False
    return m, share


def classify_subnet(
    sn, min_share=0.001, burn=0.5, wta_top1=0.5, conc_top10=0.7, min_stability=0.6
):
    """Incentive structure of one subnet, decided per snapshot, then the most common type wins."""
    m, share = working_mask(sn, min_share)
    w = np.where(m, share, 0.0)
    total = w.sum(axis=1, keepdims=True)
    s = np.divide(
        w, total, out=np.zeros_like(w), where=total > 0
    )  # shares among working miners
    n_working = m.sum(axis=1)
    top1 = s.max(axis=1)
    top10 = np.sort(s, axis=1)[:, -10:].sum(axis=1)
    burn_share = sn.inc_share[
        np.arange(sn.T), sn.owner_uid
    ]  # owner uid's share of all incentive

    per_snapshot = np.select(
        [n_working == 0, burn_share >= burn, top1 >= wta_top1, top10 >= conc_top10],
        ["no_paid_miners", "burn_dominated", "winner_take_all", "concentrated"],
        "distributed",
    )
    counts = pd.Series(per_snapshot).value_counts(normalize=True)
    mtype, stability = counts.index[0], counts.iloc[0]
    if sn.netuid == 0:
        mtype = "root"
    elif stability < min_stability:
        mtype = "mixed"  # the subnet changed structure during the period

    paid = n_working > 0
    hhi = (s**2).sum(axis=1)
    return {
        "netuid": sn.netuid,
        "mechanism_type": mtype,
        "type_stability": round(float(stability), 2),
        "main_snapshot_type": counts.index[0],
        "median_working_miners": float(np.median(n_working)),
        "median_burn_share": float(np.median(burn_share)),
        "median_top1_share": float(np.median(top1[paid])) if paid.any() else np.nan,
        "median_top10_share": float(np.median(top10[paid])) if paid.any() else np.nan,
        "median_effective_miners": (
            float(np.median(1 / hhi[paid])) if paid.any() else np.nan
        ),
        "type_breakdown": " · ".join(f"{k} {v:.0%}" for k, v in counts.items()),
    }


def classify_subnets(dataset, **kw) -> pd.DataFrame:
    return pd.DataFrame([classify_subnet(sn, **kw) for sn in dataset]).sort_values(
        ["mechanism_type", "netuid"]
    )
