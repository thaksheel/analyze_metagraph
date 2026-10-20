"""Do miners repeatedly obtain abnormal rewards without corresponding improvements in consensus?

Spike      a miner's emission share jumps >= spike_z sd above its own trailing baseline and
           by at least one fair share (1 / n_miners)
Support    change over the same interval in the median weight across validators (one
           validator one vote), the share of validators giving a fair share, and Consensus.
           Yuma clips incentive to the stake-weighted consensus, so Consensus moves with
           emission almost by construction; a reward pushed through by a few high-stake
           validators shows up as a gap to *median* support instead.
Boosters   validators whose weight on the miner jumped, and their share of StakeWeight
Repetition concentration test (Poisson with exposure) and per-miner systematic-offender
           test (binomial vs other miners in subnets of the same mechanism type)
Grouping   subnets get a mechanism type from their emission distribution, and every
           metric is also scored against peers of the same type.
"""
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from .base import Analysis

KEY_METRICS = ["rank_churn", "topk_turnover", "divergence_consensus", "divergence_median_support",
               "spikes_per_miner_day", "unsupported_share", "repeat_offender_share", "unsupported_excess_share"]


def _rolling(a: np.ndarray, L: int, fn: str) -> np.ndarray:
    return getattr(pd.DataFrame(a).rolling(L, min_periods=max(3, L // 2)), fn)().shift(1).to_numpy()


def _count_separated(times, gap: int) -> int:
    n, last = 0, -10 ** 9
    for t in times:
        if t - last >= gap:
            n, last = n + 1, t
    return n


class RewardAnomalyAnalysis(Analysis):
    key = "rewards"
    title = "Abnormal rewards without consensus improvement"
    problem = ""
    questions = {
        "Q1": "Which subnets use which incentive mechanism (winner-take-all, concentrated, ...)?",
        "Q2": "How often do reward spikes happen without broader validator support?",
        "Q3": "Do the same miners get unsupported spikes repeatedly, beyond chance?",
        "Q4": "How do churn, divergence and spikes differ by mechanism type?",
    }
    tables = {
        "events": ("Q2", "one row per reward spike: gains in emission / median support / consensus; boosting validators and their StakeWeight share"),
        "miners": ("Q3", "per miner: spikes, unsupported spikes, systematic-offender test"),
        "subnets": ("Q1-Q4", "per subnet: mechanism type, churn, divergence, spike rates, repetition tests"),
        "summary": ("Q2-Q3", "across subnets"),
        "by_mechanism": ("Q4", "medians per mechanism type"),
        "mechanism_tests": ("Q4", "Kruskal-Wallis tests of each metric across mechanism types"),
    }

    def mechanism_type(self, sn, panel) -> Dict:
        c = self.cfg.rewards
        s = sn.miner_emi_share
        live = s.sum(axis=1) > 0
        top1 = s.max(axis=1)
        top10 = np.sort(s, axis=1)[:, -10:].sum(axis=1)
        eff = 1.0 / np.maximum((s ** 2).sum(axis=1), 1e-12)
        rec = {"top1_share": float(np.median(top1[live])) if live.any() else np.nan,
               "top10_share": float(np.median(top10[live])) if live.any() else np.nan,
               "effective_miners": float(np.median(eff[live])) if live.any() else np.nan,
               "burn_share": float(panel["burn_share"].median())}
        rec["mechanism_type"] = ("burn_dominated" if rec["burn_share"] >= c.mech_burn else
                                 "winner_take_all" if rec["top1_share"] >= c.mech_wta_top1 else
                                 "concentrated" if rec["top10_share"] >= c.mech_concentrated_top10 else "distributed")
        return rec

    def per_subnet(self, sn, panel, ctx):
        c = self.cfg.rewards
        T, N = sn.T, sn.N
        mech = self.mechanism_type(sn, panel)
        sup = sn.support
        s, miners = sn.miner_emi_share, sn.is_miner
        n_m = np.maximum(miners.sum(axis=1), 1).astype(float)
        L, gap = sn.snaps(c.spike_baseline_days), sn.snaps(c.repeat_gap_days)
        mu, sdv = _rolling(s, L, "mean"), _rolling(s, L, "std")
        z = (s - mu) / np.maximum(np.nan_to_num(sdv), (c.spike_sd_floor_factor / n_m)[:, None])
        reset = pd.DataFrame(sn.replaced.astype(int)).rolling(L + 1, min_periods=1).sum().to_numpy() > 0
        spike = (z >= c.spike_z) & (s - mu >= c.spike_min_gain_factor / n_m[:, None]) & ~reset & miners & np.isfinite(mu)
        b_med, b_cons, b_br = (_rolling(x, L, "mean") for x in (sup.median_share, sup.cons_share, sup.breadth))
        m_alpha, m_tao = sn.miner_flow("alpha"), sn.miner_flow("tao")
        rows = sn.weights_rows
        events = []
        for u in np.where(spike.any(axis=0))[0]:
            ts = np.where(spike[:, u])[0]
            runs, cur = [], [ts[0]]
            for t in ts[1:]:
                if t - cur[-1] <= 2:
                    cur.append(t)
                else:
                    runs.append(cur)
                    cur = [t]
            runs.append(cur)
            for run in runs:
                t0, t1 = run[0], run[-1]
                pk = t0 + int(np.argmax(s[t0:t1 + 1, u]))
                de = float(s[pk, u] - mu[t0, u])
                dm = float(sup.median_share[pk, u] - b_med[t0, u])
                excess = np.clip(s[t0:t1 + 1, u] - mu[t0, u], 0, None)
                after = min(T - 1, t1 + L)
                # boosters: validators whose weight on u jumped vs their own baseline
                n_boost, boost_stake = 0, np.nan
                vr = sn.active_validator_rows(pk)
                if len(vr):
                    vu = rows[vr]
                    now = sn.W[pk][vr][:, u]
                    base = sn.W[max(0, t0 - L):t0][:, vr, u].mean(axis=0) if t0 > 0 else now
                    boost = (now - base) >= max(1.0 / n_m[pk], 2 * np.nan_to_num(base).mean())
                    w = sn.validator_weight(pk, vu)
                    n_boost = int(boost.sum())
                    boost_stake = float(w[boost].sum() / (w.sum() or np.nan))
                events.append({
                    "netuid": sn.netuid, "uid": int(u), "occupant": sn.occupant[t0, u],
                    "t_start": int(t0), "t_peak": int(pk), "t_end": int(t1), "hours_start": float(sn.hours[t0]),
                    "duration_hours": float(sn.hours[t1] - sn.hours[t0]), "z_peak": float(z[pk, u]),
                    "emission_gain": de, "median_support_gain": dm,
                    "consensus_gain": float(sup.cons_share[pk, u] - b_cons[t0, u]),
                    "breadth_gain": float(sup.breadth[pk, u] - b_br[t0, u]),
                    "support_ratio": dm / de if de > 0 else np.nan,
                    "divergence_consensus_peak": float(s[pk, u] - sup.cons_share[pk, u]),
                    "divergence_support_peak": float(s[pk, u] - sup.median_share[pk, u]),
                    "unsupported": bool(de > 0 and dm < c.support_ratio * de),
                    "sustained": bool(sn.occupant[after, u] == sn.occupant[t0, u] and s[after, u] - mu[t0, u] >= 0.5 * de),
                    "boosting_validators": n_boost, "boosting_stake_share": boost_stake,
                    "excess_alpha": float((excess * m_alpha[t0:t1 + 1]).sum()),
                    "excess_tao": float((excess * m_tao[t0:t1 + 1]).sum())})
        ev = pd.DataFrame(events)

        # per-miner repetition
        occ_m = pd.Series(sn.occupant[miners])
        exposure = occ_m.value_counts()
        mean_sh = pd.DataFrame({"occ": sn.occupant[miners], "s": s[miners]}).groupby("occ")["s"].mean()
        relevant = exposure[(exposure >= 2 * L) & (mean_sh.reindex(exposure.index) > 0)]
        counts = pd.Series(0.0, index=relevant.index)
        mrows = []
        if len(ev):
            for occ, g in ev.sort_values("t_start").groupby("occupant"):
                uns = g[g["unsupported"]]
                k = _count_separated(uns["t_start"], gap)
                if occ in counts.index:
                    counts[occ] = k
                mrows.append({"netuid": sn.netuid, "occupant": occ, "spikes": _count_separated(g["t_start"], gap),
                              "unsupported_spikes": k, "exposure_snapshots": int(exposure.get(occ, 0)),
                              "unsupported_excess_alpha": float(uns["excess_alpha"].sum()),
                              "repeat_offender": k >= c.repeat_min})
        mdf = pd.DataFrame(mrows)
        conc_p = np.nan
        if counts.sum() >= 5 and len(counts) >= 5:
            e = counts.sum() / relevant.sum() * relevant.astype(float)
            conc_p = float(stats.chi2.sf(float((((counts - e) ** 2) / e).sum()), len(counts) - 1))

        # churn and divergence
        pct = np.zeros((T, N))
        for t in range(T):
            m = miners[t]
            if m.sum() > 1:
                pct[t, m] = (stats.rankdata(s[t, m]) - 1) / (m.sum() - 1)
        same = miners[1:] & miners[:-1] & ~sn.replaced[1:]
        churn = [np.abs(pct[t + 1, same[t]] - pct[t, same[t]]).mean() for t in range(T - 1) if same[t].any()]
        k = c.churn_top_k
        turn = [1 - len(set(sn.occupant[t - 1][np.argsort(-s[t - 1])[:k]]) & set(sn.occupant[t][np.argsort(-s[t])[:k]])) / k
                for t in range(1, T)]
        alpha_total = float(sn.alpha_flow.sum()) or np.nan
        rep = mdf[mdf["repeat_offender"]] if len(mdf) else mdf
        uns_ex = float(ev.loc[ev["unsupported"], "excess_alpha"].sum()) if len(ev) else 0.0
        summ = {"netuid": sn.netuid, **mech,
                "rank_churn": float(np.mean(churn)) if churn else np.nan,
                "topk_turnover": float(np.mean(turn)) if turn else np.nan,
                "divergence_consensus": float(np.nanmean(0.5 * np.abs(s - sup.cons_share).sum(axis=1))),
                "divergence_median_support": float(np.nanmean(0.5 * np.abs(s - sup.median_share).sum(axis=1))),
                "spikes": len(ev), "unsupported_spikes": int(ev["unsupported"].sum()) if len(ev) else 0,
                "spikes_per_miner_day": len(ev) / (max(len(relevant), 1) * max(sn.period_days, 1e-9)),
                "unsupported_share": float(ev["unsupported"].mean()) if len(ev) else np.nan,
                "sustained_share": float(ev["sustained"].mean()) if len(ev) else np.nan,
                "median_boosting_stake_share_unsupported": float(ev.loc[ev["unsupported"], "boosting_stake_share"].median())
                if len(ev) and ev["unsupported"].any() else np.nan,
                "repeat_offenders": len(rep), "repeat_offender_share": len(rep) / max(len(relevant), 1),
                "unsupported_excess_alpha": uns_ex,
                "unsupported_excess_tao": float(ev.loc[ev["unsupported"], "excess_tao"].sum()) if len(ev) else 0.0,
                "unsupported_excess_share": uns_ex / alpha_total,
                "concentration_p_value": conc_p,
                "emission_vs_consensus_gain_corr": float(stats.spearmanr(ev["emission_gain"], ev["consensus_gain"]).statistic)
                if len(ev) >= 5 and ev["consensus_gain"].nunique() > 1 else np.nan}
        return {"events": ev, "miners": mdf, "subnets": pd.DataFrame([summ])}

    def aggregate(self, results, panel):
        c = self.cfg.rewards
        s, mdf, ev = self.combined("subnets").copy(), self.combined("miners"), self.combined("events")
        if not len(s):
            return {}
        s["systematic_offenders"], s["systematic_share_of_unsupported_excess"] = 0, 0.0
        if len(mdf):
            mdf = mdf.merge(s[["netuid", "mechanism_type"]], on="netuid", how="left")
            tot = mdf.groupby("mechanism_type")[["spikes", "unsupported_spikes"]].sum()
            pv = []
            for _, r in mdf.iterrows():
                t = tot.loc[r["mechanism_type"]]
                p0 = (t["unsupported_spikes"] - r["unsupported_spikes"] + 1) / (t["spikes"] - r["spikes"] + 2)
                n = int(max(r["spikes"], r["unsupported_spikes"]))
                pv.append(float(stats.binomtest(int(r["unsupported_spikes"]), n, p0, alternative="greater").pvalue) if n else np.nan)
            mdf["systematic_p_value"] = pv
            mdf["systematic_offender"] = (mdf["systematic_p_value"] < c.systematic_p) & (mdf["unsupported_spikes"] >= c.repeat_min)
            s["systematic_offenders"] = s["netuid"].map(mdf.groupby("netuid")["systematic_offender"].sum()).fillna(0).astype(int)
            sx = mdf[mdf["systematic_offender"]].groupby("netuid")["unsupported_excess_alpha"].sum()
            s["systematic_share_of_unsupported_excess"] = (s["netuid"].map(sx).fillna(0) /
                                                           s["unsupported_excess_alpha"].replace(0, np.nan)).fillna(0.0)
        for m in KEY_METRICS:
            g = s.groupby("mechanism_type")[m]
            mad = g.transform(lambda x: (x - x.median()).abs().median() * 1.4826)
            s[f"{m}_type_z"] = (s[m] - g.transform("median")) / mad.replace(0, np.nan)
        by = s.groupby("mechanism_type").agg(subnets=("netuid", "count"),
                                             **{f"median_{m}": (m, "median") for m in KEY_METRICS},
                                             subnets_with_systematic_offenders=("systematic_offenders", lambda x: int((x > 0).sum()))
                                             ).reset_index()
        tests = []
        groups = [g for _, g in s.groupby("mechanism_type") if len(g) >= 2]
        for m in KEY_METRICS:
            vals = [g[m].dropna() for g in groups if g[m].notna().sum() >= 2]
            if len(vals) >= 2 and len({float(x) for v in vals for x in v}) > 1:
                h = stats.kruskal(*vals)
                tests.append({"metric": m, "groups": len(vals), "kruskal_h": float(h.statistic), "p_value": float(h.pvalue)})
        summ = pd.DataFrame([{
            "subnets": len(s),
            "mechanism_types": ", ".join(f"{k}={v}" for k, v in s["mechanism_type"].value_counts().items()),
            "spikes": int(s["spikes"].sum()),
            "share_spikes_unsupported": float(ev["unsupported"].mean()) if len(ev) else np.nan,
            "median_boosting_stake_share_unsupported": float(ev.loc[ev["unsupported"], "boosting_stake_share"].median())
            if len(ev) and ev["unsupported"].any() else np.nan,
            "share_subnets_concentrated": float((s["concentration_p_value"] < 0.05).mean()),
            "share_subnets_with_systematic_offenders": float((s["systematic_offenders"] > 0).mean()),
            "systematic_offenders": int(s["systematic_offenders"].sum()),
            "total_unsupported_excess_tao": float(s["unsupported_excess_tao"].sum()),
            "median_emission_vs_consensus_gain_corr": float(s["emission_vs_consensus_gain_corr"].median())}])
        # replace the per-subnet tables with the enriched versions
        return {"subnets": s, "miners": mdf, "summary": summ, "by_mechanism": by, "mechanism_tests": pd.DataFrame(tests)}
