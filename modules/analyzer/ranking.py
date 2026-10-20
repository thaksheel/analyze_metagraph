"""Turn area results into a ranked problem list.

Impact is expressed as % of network emission affected over the sample period:
TAO-weighted when SubnetTAO/SubnetAlphaIn were collected, otherwise the mean
across subnets of the affected share of each subnet's own emission.
Perceived importance and solving ease are not measurable on-chain; they are
carried through from the input table as a second axis.
"""
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config

PROBLEMS = {
    "A1": "miner adaptation exploits",
    "A2": "insufficient miner-side telemetry",
    "A3": "understanding mvs interactions",
    "A4": "need for actionable metagraph insights",
    "A5": "sybil attack vulnerability",
}
HYPOTHESES = {
    "A1": "exploit cycles every 2-4 weeks",
    "A2": "<10% of subnets collect miner-level signals",
    "A3": "mis-scoring leads to poor miner selection",
    "A4": ">50 raw signals, few actionable",
    "A5": "15-25% of miners sybil-like in some subnets",
}


def _band(v: float, edges) -> int:
    return int(1 + sum(v >= e for e in edges)) if np.isfinite(v) else np.nan


def _sum_by(df, col):
    if df is None or not len(df) or col not in df:
        return pd.Series(dtype=float)
    return df.groupby("netuid")[col].sum()


def rank(R: Dict[str, pd.DataFrame], cfg: Config) -> Dict[str, pd.DataFrame]:
    subs = R["a1_q1_subnets"].set_index("netuid")
    tot_tao = subs["subnet_tao_flow_total"]
    tot_alpha = subs["subnet_alpha_flow_total"].replace(0, np.nan)
    use_tao = float(tot_tao.sum()) > 0
    n = len(subs)

    def impact_pct(alpha_by_net: pd.Series, tao_by_net: pd.Series) -> float:
        if use_tao:
            return float(100 * tao_by_net.reindex(subs.index).fillna(0).sum() / tot_tao.sum())
        return float(100 * (alpha_by_net.reindex(subs.index).fillna(0) / tot_alpha).mean())

    rows = []

    # A1: at-risk emission, minus the part also counted as sybil capture
    inc = R.get("a1_incidents", pd.DataFrame())
    coh = R.get("a1_cohort_members", pd.DataFrame())
    mem = R.get("a5_members", pd.DataFrame())
    overlap = 0.0
    if len(inc):
        inc = inc.copy()
        inc["overlap_frac"] = 0.0
        if len(coh) and len(mem):
            syb = set(zip(mem["netuid"], mem["occupant"]))
            coh = coh.copy()
            coh["in_sybil"] = [(a, b) in syb for a, b in zip(coh["netuid"], coh["occupant"])]
            g = coh.groupby("incident_id").apply(
                lambda d: d.loc[d["in_sybil"], "share_gain"].sum() / (d["share_gain"].sum() or np.nan),
                include_groups=False)
            inc["overlap_frac"] = inc["incident_id"].map(g).fillna(0.0)
        overlap = float(inc["overlap_frac"].mean())
        inc["risk_alpha_dedup"] = inc["at_risk_alpha"] * (1 - inc["overlap_frac"])
        inc["risk_tao_dedup"] = inc["at_risk_tao"] * (1 - inc["overlap_frac"])
    a1_imp = impact_pct(_sum_by(inc, "risk_alpha_dedup"), _sum_by(inc, "risk_tao_dedup"))
    a1q1 = R["a1_q1_summary"].iloc[0]
    rows.append({"id": "A1", "size_metric": "share of subnets with recurring regime shifts",
                 "size_value": a1q1["share_subnets_recurring"], "impact_pct": a1_imp,
                 "measured": f"median interval {a1q1['median_interval_days']:.1f} d; "
                             f"{int(a1q1['total_incidents'])} incidents; sybil overlap {overlap:.0%} removed"})

    # A2: excess incident burden in low-telemetry subnets
    a2 = R["a2_q1_subnets"].set_index("netuid")
    a2s = R["a2_q3_summary"].iloc[0]
    raw_gap = float(a2s["excess_burden_low_vs_high"])
    supported = bool(a2s["evidence_supported"])
    # only attribute impact to missing telemetry when the low-vs-high gap is significant
    gap = max(raw_gap, 0.0) if (supported and np.isfinite(raw_gap)) else 0.0
    q3s = R["a2_q3_subnets"].set_index("netuid")
    ti = q3s["telemetry_index"]
    low = q3s.index[ti <= ti.median()] if ti.nunique() > 1 else q3s.index
    a2_alpha = pd.Series(gap, index=low) * tot_alpha.reindex(low)
    a2_tao = pd.Series(gap, index=low) * tot_tao.reindex(low)
    no_sig = 1 - float(a2["any_onchain_miner_signal"].mean())
    rows.append({"id": "A2", "size_metric": "share of subnets with no on-chain miner signal",
                 "size_value": no_sig, "impact_pct": impact_pct(a2_alpha, a2_tao),
                 "measured": f"{1 - no_sig:.0%} of subnets show on-chain miner signals; "
                             f"low-vs-high burden gap {raw_gap:.2%} of subnet emission "
                             f"(p={a2s['p_value_low_greater']:.2g}, "
                             f"{'counted' if supported else 'not significant or too few subnets; impact set to 0'})"})

    # A3: contested emission scaled by disagreement (conservative)
    q1 = R["a3_q1_subnets"].set_index("netuid")
    q3 = R["a3_q3_subnets"].set_index("netuid")
    dis = (1 - q1["validator_agreement"]).clip(lower=0).fillna(0)
    a3_alpha = q3["contested_alpha"] * dis
    a3_tao = q3["contested_tao"] * dis
    upper = impact_pct(q3["contested_alpha"], q3["contested_tao"])
    rows.append({"id": "A3", "size_metric": "share of subnets with low validator agreement",
                 "size_value": float(q1["low_agreement"].mean()), "impact_pct": impact_pct(a3_alpha, a3_tao),
                 "measured": f"upper bound (all contested emission) {upper:.2f}%; median replace-rate gap "
                             f"{R['a3_summary'].iloc[0]['median_replace_rate_gap']:.3f}"})

    # A4: detection-gap cost
    gapdf = R.get("a4_q3_incidents", pd.DataFrame())
    if len(gapdf):
        helped = gapdf[(gapdf["gap_hours"] > 0) | (~gapdf["responded"])]
        a4_size = helped["netuid"].nunique() / n
    else:
        a4_size = 0.0
    a4q1 = R["a4_q1_summary"].iloc[0]
    rows.append({"id": "A4", "size_metric": "share of subnets where earlier detection was possible",
                 "size_value": a4_size,
                 "impact_pct": impact_pct(_sum_by(gapdf, "gap_cost_alpha"), _sum_by(gapdf, "gap_cost_tao")),
                 "measured": f"unresponded incidents add up to "
                             f"{impact_pct(_sum_by(gapdf, 'unresponded_at_risk_alpha'), _sum_by(gapdf, 'unresponded_at_risk_tao')):.2f}% (upper bound); "
                             f"{int(a4q1['raw_storage_fields'])} raw fields, {int(a4q1['derived_signals_used'])} derived; "
                             f"{int(a4q1['components_80pct'])} components explain 80%; "
                             f"{int(R['a4_q2_summary'].iloc[0]['actionable_signals'])} actionable"})

    # A5: excess emission captured by clusters
    clu = R.get("a5_clusters", pd.DataFrame())
    a5s = R["a5_q1_summary"].iloc[0]
    rows.append({"id": "A5", "size_metric": f"share of subnets with >= {cfg.sybil_share_threshold:.0%} clustered miners",
                 "size_value": a5s["share_subnets_above_threshold"],
                 "impact_pct": impact_pct(_sum_by(clu, "excess_alpha"), _sum_by(clu, "excess_tao")),
                 "measured": f"median cluster share {a5s['median_cluster_share']:.1%}, p90 {a5s['p90_cluster_share']:.1%}"})

    df = pd.DataFrame(rows)
    df["problem"] = df["id"].map(PROBLEMS)
    df["hypothesis"] = df["id"].map(HYPOTHESES)
    df["size_score"] = df["size_value"].apply(lambda v: _band(v, cfg.size_bands))
    df["impact_score"] = df["impact_pct"].apply(lambda v: _band(v, cfg.impact_bands_pct))
    df["priority"] = df["size_score"] * df["impact_score"]
    df["impact_unit"] = "% of network TAO emission" if use_tao else "% of avg subnet emission (alpha)"
    df = df.sort_values(["priority", "impact_pct"], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    cols = ["rank", "id", "problem", "priority", "size_score", "impact_score", "size_metric",
            "size_value", "impact_pct", "impact_unit", "hypothesis", "measured"]
    return {"ranking": df[cols]}


def add_b_evidence(R: Dict[str, pd.DataFrame]) -> None:
    """Attach the B1-B3 headline numbers to the related problems in the ranking."""
    if "ranking" not in R:
        return
    ev = {}
    b1 = R.get("b1_summary")
    if b1 is not None and len(b1):
        r = b1.iloc[0]
        txt = (f"B1: sybil-like tracks in {r['share_subnets_with_sybil_like_track']:.0%} of subnets, "
               f"p90 member share {r['p90_sybil_member_share']:.1%}")
        if "precision_identity_confirmed" in r and np.isfinite(r["precision_identity_confirmed"]):
            txt += f", identity-confirmed precision {r['precision_identity_confirmed']:.0%}"
        ev["A5"] = txt
    b2 = R.get("b2_summary")
    if b2 is not None and len(b2):
        r = b2.iloc[0]
        ev["A1"] = (f"B2: {r['share_spikes_unsupported']:.0%} of reward spikes lack broad validator support; "
                    f"systematic offenders in {r['share_subnets_with_systematic_offenders']:.0%} of subnets")
        ev["A3"] = ev["A1"]
    b3 = R.get("b3_verdict")
    if b3 is not None and len(b3):
        used = b3[~b3["verdict"].str.startswith("not used")]
        txt = "B3: out-of-sample R² " + ", ".join(
            f"{t} {v:.2f} ({w})" for t, v, w in zip(used["target"], used["best_r2_all_signals"], used["verdict"]))
        ev["A2"] = txt
        ev["A4"] = txt
    R["ranking"]["supporting_evidence"] = R["ranking"]["id"].map(ev).fillna("")
