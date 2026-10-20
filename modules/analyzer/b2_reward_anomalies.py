"""B2 — Do miners repeatedly obtain abnormal rewards without corresponding
improvements in consensus?

Per miner:
  spike      emission share jumps >= spike_z robust-ish sd above its own trailing
             baseline AND by at least one fair share (1 / n_miners)
  support    how much validator support rose over the same interval, measured
             one-validator-one-vote (median weight across validators) and as
             breadth (share of validators giving a fair share); Consensus storage
             change is recorded too
  unsupported spike = support gain < support_ratio * emission gain

Per subnet:
  rank churn (mean |change in percentile rank|, top-k turnover)
  emission-consensus divergence (vs Consensus storage and vs median support)
  repetition test: are unsupported spikes concentrated in the same miners more
  than chance? (Poisson dispersion test with per-miner exposure)

Subnets are grouped by incentive mechanism type, inferred from the emission
distribution (burn-dominated, winner-take-all, concentrated, distributed), and each
subnet is also scored relative to the other subnets of its type, since rank churn
and spikes mean different things in a winner-take-all subnet.
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config
from .subnet import SubnetData
from .support import validator_support

KEY_METRICS = [
    "rank_churn",
    "topk_turnover",
    "divergence_consensus",
    "divergence_median_support",
    "spikes_per_miner_day",
    "unsupported_share",
    "repeat_offender_share",
    "unsupported_excess_share",
]


def _count_separated(times, gap: int) -> int:
    n, last = 0, -(10**9)
    for t in times:
        if t - last >= gap:
            n += 1
            last = t
    return n


def mechanism_type(sd: SubnetData, panel: pd.DataFrame, cfg: Config) -> Dict:
    s = sd.miner_emi_share
    live = s.sum(axis=1) > 0
    top1 = s.max(axis=1)
    top10 = np.sort(s, axis=1)[:, -10:].sum(axis=1)
    eff = 1.0 / np.maximum((s**2).sum(axis=1), 1e-12)
    rec = {
        "top1_share": float(np.median(top1[live])) if live.any() else np.nan,
        "top10_share": float(np.median(top10[live])) if live.any() else np.nan,
        "effective_miners": float(np.median(eff[live])) if live.any() else np.nan,
        "burn_share": float(panel["burn_share"].median()),
    }
    if rec["burn_share"] >= cfg.mech_burn:
        t = "burn_dominated"
    elif rec["top1_share"] >= cfg.mech_wta_top1:
        t = "winner_take_all"
    elif rec["top10_share"] >= cfg.mech_concentrated_top10:
        t = "concentrated"
    else:
        t = "distributed"
    rec["mechanism_type"] = t
    return rec


def _rolling(a: np.ndarray, L: int, fn: str) -> np.ndarray:
    df = pd.DataFrame(a)
    r = getattr(df.rolling(L, min_periods=max(3, L // 2)), fn)()
    return r.shift(1).to_numpy()


def per_subnet(
    sd: SubnetData, panel: pd.DataFrame, cfg: Config
) -> Dict[str, pd.DataFrame]:
    T, N = sd.T, sd.N
    mech = mechanism_type(sd, panel, cfg)
    sup = validator_support(sd)
    s = sd.miner_emi_share
    miners = sd.is_miner
    n_m = np.maximum(miners.sum(axis=1), 1).astype(float)
    L = sd.snaps(cfg.spike_baseline_days)
    gap = sd.snaps(cfg.repeat_gap_days)

    # ---- spikes ----
    mu = _rolling(s, L, "mean")
    sdv = _rolling(s, L, "std")
    floor = cfg.spike_sd_floor_factor / n_m
    z = (s - mu) / np.maximum(np.nan_to_num(sdv), floor[:, None])
    gain = s - mu
    reset = (
        pd.DataFrame(sd.replaced.astype(int))
        .rolling(L + 1, min_periods=1)
        .sum()
        .to_numpy()
        > 0
    )
    spike = (
        (z >= cfg.spike_z)
        & (gain >= cfg.spike_min_gain_factor / n_m[:, None])
        & ~reset
        & miners
    )
    spike &= np.isfinite(mu)
    base_med = _rolling(sup.median_share, L, "mean")
    base_cons = _rolling(sup.cons_share, L, "mean")
    base_br = _rolling(sup.breadth, L, "mean")
    miner_alpha = np.where(miners, sd.alpha_flow, 0).sum(axis=1)
    miner_tao = np.where(miners, sd.tao_flow, 0).sum(axis=1)

    events = []
    for u in np.where(spike.any(axis=0))[0]:
        ts = np.where(spike[:, u])[0]
        runs, cur = [], [ts[0]]
        for t in ts[1:]:
            if t - cur[-1] <= 2:  # allow a one-snapshot gap inside a run
                cur.append(t)
            else:
                runs.append(cur)
                cur = [t]
        runs.append(cur)
        for run in runs:
            t0, t1 = run[0], run[-1]
            pk = t0 + int(np.argmax(s[t0 : t1 + 1, u]))
            de = float(s[pk, u] - mu[t0, u])
            dm = float(sup.median_share[pk, u] - base_med[t0, u])
            dc = float(sup.cons_share[pk, u] - base_cons[t0, u])
            db = float(sup.breadth[pk, u] - base_br[t0, u])
            excess = np.clip(s[t0 : t1 + 1, u] - mu[t0, u], 0, None)
            after = min(T - 1, t1 + L)
            same_after = sd.occupant[after, u] == sd.occupant[t0, u]
            events.append(
                {
                    "netuid": sd.netuid,
                    "uid": int(u),
                    "occupant": sd.occupant[t0, u],
                    "t_start": int(t0),
                    "t_peak": int(pk),
                    "t_end": int(t1),
                    "hours_start": float(sd.hours[t0]),
                    "duration_hours": float(sd.hours[t1] - sd.hours[t0]),
                    "z_peak": float(z[pk, u]),
                    "emission_gain": de,
                    "median_support_gain": dm,
                    "consensus_gain": dc,
                    "breadth_gain": db,
                    "support_ratio": dm / de if de > 0 else np.nan,
                    "consensus_ratio": dc / de if de > 0 else np.nan,
                    "divergence_consensus_peak": float(
                        s[pk, u] - sup.cons_share[pk, u]
                    ),
                    "divergence_support_peak": float(
                        s[pk, u] - sup.median_share[pk, u]
                    ),
                    "unsupported": bool(de > 0 and dm < cfg.support_ratio * de),
                    "sustained": bool(
                        same_after and s[after, u] - mu[t0, u] >= 0.5 * de
                    ),
                    "excess_alpha": float((excess * miner_alpha[t0 : t1 + 1]).sum()),
                    "excess_tao": float((excess * miner_tao[t0 : t1 + 1]).sum()),
                }
            )
    ev = pd.DataFrame(events)

    # ---- per-miner repetition ----
    occ_present = pd.DataFrame({"occ": sd.occupant[miners], "uid": np.where(miners)[1]})
    exposure = occ_present.groupby("occ").size()
    rel_share = (
        pd.DataFrame({"occ": sd.occupant[miners], "s": s[miners]})
        .groupby("occ")["s"]
        .mean()
    )
    relevant = exposure[(exposure >= 2 * L) & (rel_share.reindex(exposure.index) > 0)]
    counts = pd.Series(0, index=relevant.index, dtype=float)
    miner_rows = []
    if len(ev):
        for occ, g in ev.sort_values("t_start").groupby("occupant"):
            uns = g[g["unsupported"]]
            n_uns = _count_separated(uns["t_start"], gap)
            if occ in counts.index:
                counts[occ] = n_uns
            miner_rows.append(
                {
                    "netuid": sd.netuid,
                    "occupant": occ,
                    "spikes": _count_separated(g["t_start"], gap),
                    "unsupported_spikes": n_uns,
                    "exposure_snapshots": int(exposure.get(occ, 0)),
                    "unsupported_excess_alpha": float(uns["excess_alpha"].sum()),
                    "repeat_offender": n_uns >= cfg.repeat_min,
                }
            )
    mdf = pd.DataFrame(miner_rows)

    disp_p = np.nan
    if counts.sum() >= 5 and len(counts) >= 5:
        rate = counts.sum() / relevant.sum()
        e = rate * relevant.astype(float)
        chi2 = float((((counts - e) ** 2) / e).sum())
        disp_p = float(stats.chi2.sf(chi2, len(counts) - 1))

    # ---- subnet-level churn and divergence ----
    pct = np.zeros((T, N))
    for t in range(T):
        m = miners[t]
        if m.sum() > 1:
            pct[t, m] = (stats.rankdata(s[t, m]) - 1) / (m.sum() - 1)
    same = miners[1:] & miners[:-1] & ~sd.replaced[1:]
    churn = np.array(
        [
            (
                np.abs(pct[t + 1, same[t]] - pct[t, same[t]]).mean()
                if same[t].any()
                else np.nan
            )
            for t in range(T - 1)
        ]
    )
    k = cfg.churn_top_k
    turn = []
    for t in range(1, T):
        a = set(sd.occupant[t - 1][np.argsort(-s[t - 1])[:k]])
        b = set(sd.occupant[t][np.argsort(-s[t])[:k]])
        turn.append(1 - len(a & b) / k)
    div_c = 0.5 * np.abs(s - sup.cons_share).sum(axis=1)
    div_m = 0.5 * np.abs(s - sup.median_share).sum(axis=1)
    days = max((sd.blocks[-1] - sd.blocks[0]) * 12 / 86400, 1e-9)
    alpha_total = float(sd.alpha_flow.sum()) or np.nan
    n_unsup = int(ev["unsupported"].sum()) if len(ev) else 0
    rep = mdf[mdf["repeat_offender"]] if len(mdf) else mdf
    uns_ex = float(ev.loc[ev["unsupported"], "excess_alpha"].sum()) if len(ev) else 0.0
    summ = {
        "netuid": sd.netuid,
        **mech,
        "rank_churn": float(np.nanmean(churn)),
        "topk_turnover": float(np.mean(turn)) if turn else np.nan,
        "divergence_consensus": float(np.nanmean(div_c)),
        "divergence_median_support": float(np.nanmean(div_m)),
        "spikes": len(ev),
        "unsupported_spikes": n_unsup,
        "spikes_per_miner_day": len(ev) / (max(len(relevant), 1) * days),
        "unsupported_share": n_unsup / len(ev) if len(ev) else np.nan,
        "sustained_share": float(ev["sustained"].mean()) if len(ev) else np.nan,
        "repeat_offenders": len(rep),
        "repeat_offender_share": len(rep) / max(len(relevant), 1),
        "unsupported_excess_share": uns_ex / alpha_total,
        "unsupported_excess_alpha": uns_ex,
        "unsupported_excess_tao": (
            float(ev.loc[ev["unsupported"], "excess_tao"].sum()) if len(ev) else 0.0
        ),
        "repeat_share_of_unsupported_excess": (
            float(rep["unsupported_excess_alpha"].sum() / uns_ex)
            if uns_ex > 0 and len(rep)
            else 0.0
        ),
        "concentration_p_value": disp_p,
        "emission_vs_consensus_gain_corr": (
            float(stats.spearmanr(ev["emission_gain"], ev["consensus_gain"]).statistic)
            if len(ev) >= 5 and ev["consensus_gain"].nunique() > 1
            else np.nan
        ),
    }
    return {"b2_events": ev, "b2_miners": mdf, "b2_subnets": pd.DataFrame([summ])}


def aggregate(R: Dict[str, pd.DataFrame], cfg: Config) -> Dict[str, pd.DataFrame]:
    s = R.get("b2_subnets", pd.DataFrame())
    if not len(s):
        return {}
    s = s.copy()
    # systematic offenders: a miner whose spikes are unsupported far more often than
    # spikes of other miners in subnets of the same mechanism type (binomial test)
    mdf = R.get("b2_miners", pd.DataFrame())
    s["systematic_offenders"] = 0
    s["systematic_share_of_unsupported_excess"] = 0.0
    if len(mdf):
        mdf = mdf.merge(s[["netuid", "mechanism_type"]], on="netuid", how="left")
        tot = mdf.groupby("mechanism_type")[["spikes", "unsupported_spikes"]].sum()
        p_vals = []
        for _, r in mdf.iterrows():
            t = tot.loc[r["mechanism_type"]]
            n_o, k_o = (
                t["spikes"] - r["spikes"],
                t["unsupported_spikes"] - r["unsupported_spikes"],
            )
            p0 = (k_o + 1) / (n_o + 2)  # other miners' rate, with a weak prior
            p_vals.append(
                float(
                    stats.binomtest(
                        int(r["unsupported_spikes"]),
                        int(max(r["spikes"], r["unsupported_spikes"])),
                        p0,
                        alternative="greater",
                    ).pvalue
                )
                if r["spikes"] > 0
                else np.nan
            )
        mdf["systematic_p_value"] = p_vals
        mdf["systematic_offender"] = (mdf["systematic_p_value"] < 0.01) & (
            mdf["unsupported_spikes"] >= cfg.repeat_min
        )
        R["b2_miners"] = mdf
        g = mdf.groupby("netuid")
        s["systematic_offenders"] = (
            s["netuid"].map(g["systematic_offender"].sum()).fillna(0).astype(int)
        )
        sys_ex = (
            mdf[mdf["systematic_offender"]]
            .groupby("netuid")["unsupported_excess_alpha"]
            .sum()
        )
        s["systematic_share_of_unsupported_excess"] = (
            s["netuid"].map(sys_ex).fillna(0)
            / s["unsupported_excess_alpha"].replace(0, np.nan)
        ).fillna(0.0)
    # score each subnet against peers of the same mechanism type
    for m in KEY_METRICS:
        g = s.groupby("mechanism_type")[m]
        med = g.transform("median")
        mad = g.transform(lambda x: (x - x.median()).abs().median() * 1.4826)
        s[f"{m}_type_z"] = (s[m] - med) / mad.replace(0, np.nan)
    R["b2_subnets"] = s

    by = (
        s.groupby("mechanism_type")
        .agg(
            subnets=("netuid", "count"),
            **{f"median_{m}": (m, "median") for m in KEY_METRICS},
            subnets_with_repeat_offenders=(
                "repeat_offenders",
                lambda x: int((x > 0).sum()),
            ),
            subnets_concentrated=(
                "concentration_p_value",
                lambda x: int((x < 0.05).sum()),
            ),
            subnets_with_systematic_offenders=(
                "systematic_offenders",
                lambda x: int((x > 0).sum()),
            ),
        )
        .reset_index()
    )

    tests = []
    groups = [g for _, g in s.groupby("mechanism_type") if len(g) >= 2]
    for m in KEY_METRICS:
        vals = [g[m].dropna() for g in groups]
        vals = [v for v in vals if len(v) >= 2]
        if len(vals) >= 2 and len({float(x) for v in vals for x in v}) > 1:
            h = stats.kruskal(*vals)
            tests.append(
                {
                    "metric": m,
                    "groups": len(vals),
                    "kruskal_h": float(h.statistic),
                    "p_value": float(h.pvalue),
                }
            )

    ev = R.get("b2_events", pd.DataFrame())
    summ = pd.DataFrame(
        [
            {
                "subnets": len(s),
                "mechanism_types": ", ".join(
                    f"{k}={v}" for k, v in s["mechanism_type"].value_counts().items()
                ),
                "spikes": int(s["spikes"].sum()),
                "share_spikes_unsupported": (
                    float(ev["unsupported"].mean()) if len(ev) else np.nan
                ),
                "share_subnets_with_repeat_offenders": float(
                    (s["repeat_offenders"] > 0).mean()
                ),
                "share_subnets_concentrated": float(
                    (s["concentration_p_value"] < 0.05).mean()
                ),
                "share_subnets_with_systematic_offenders": float(
                    (s["systematic_offenders"] > 0).mean()
                ),
                "systematic_offenders": int(s["systematic_offenders"].sum()),
                "median_systematic_share_of_unsupported_excess": float(
                    s["systematic_share_of_unsupported_excess"].median()
                ),
                "median_unsupported_excess_share": float(
                    s["unsupported_excess_share"].median()
                ),
                "total_unsupported_excess_tao": float(
                    s["unsupported_excess_tao"].sum()
                ),
                "median_repeat_share_of_unsupported_excess": float(
                    s["repeat_share_of_unsupported_excess"].median()
                ),
                "median_emission_vs_consensus_gain_corr": float(
                    s["emission_vs_consensus_gain_corr"].median()
                ),
            }
        ]
    )
    return {
        "b2_summary": summ,
        "b2_by_mechanism": by,
        "b2_mechanism_tests": pd.DataFrame(tests),
    }
