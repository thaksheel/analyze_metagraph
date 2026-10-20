"""Data checks to run before trusting any result (see plan, 'Data checks')."""
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from .subnet import Dataset


def run_checks(ds: Dataset) -> Dict[str, pd.DataFrame]:
    meta = ds.meta
    rows = []
    last_alpha, last_tao = {}, {}
    for sd in ds:
        n_v_rows = len(sd.weights_rows)
        permits = np.nanmax(np.nansum(np.nan_to_num(sd.vpermit), axis=1)) if sd.T else 0
        flows = sd.alpha_flow
        rows.append({
            "netuid": sd.netuid,
            "snapshots": sd.T,
            "median_spacing_blocks": sd.spacing_blocks,
            "max_gap_blocks": int(np.max(np.diff(sd.blocks))) if sd.T > 1 else 0,
            "uid_count_changes": int((np.diff(sd.n_uids) != 0).sum()),
            "max_uids": sd.N,
            "max_validator_permits": int(permits),
            "weight_rows_seen": n_v_rows,
            "weights_look_complete": bool(n_v_rows >= 0.5 * permits) if permits else False,
            "bond_rows_seen": len(sd.bonds_rows),
            "mechanism_count_max": float(np.nanmax(sd.mechanism_count)) if np.isfinite(sd.mechanism_count).any() else np.nan,
            "zero_emission_snapshots": int((np.nansum(sd.emission, axis=1) == 0).sum()),
            "has_hotkeys": sd.has_hotkeys,
            "has_reg_block": sd.has_reg_block,
            "has_price": sd.has_price,
            "has_owner_hotkey": bool((sd.owner_hotkey != "").any()),
            "has_yc3": bool(np.isfinite(sd.yc3).any()),
            "has_identity": bool(sd.identity),
            "has_coldkeys": bool((sd.coldkeys != "").any()),
            "has_axons": bool(np.isfinite(sd.axon_block).any()),
            "has_commitments": bool(np.isfinite(sd.commit_block).any()),
            "has_stake": bool(np.isfinite(sd.stake).any()),
            "commit_reveal_on": bool(np.nanmax(sd.commit_reveal) > 0) if np.isfinite(sd.commit_reveal).any() else False,
        })
        last_alpha[sd.netuid] = float(flows[-1].sum())
        last_tao[sd.netuid] = float(sd.tao_flow[-1].sum())
    df = pd.DataFrame(rows)

    issues = []
    if (df["mechanism_count_max"] > 1).any():
        nets = df.loc[df["mechanism_count_max"] > 1, "netuid"].tolist()
        issues.append(f"Subnets with >1 mechanism (confirm which mechanism's weights were stored): {nets}")
    if (~df["weights_look_complete"]).any():
        nets = df.loc[~df["weights_look_complete"], "netuid"].tolist()
        issues.append(f"Weights have far fewer rows than validator permits (partial-key query?): {nets}")
    for col, label in [("has_hotkeys", "Keys"), ("has_reg_block", "BlockAtRegistration"),
                       ("has_price", "SubnetTAO/SubnetAlphaIn"), ("has_owner_hotkey", "SubnetOwnerHotkey"),
                       ("has_yc3", "Yuma3On"), ("has_identity", "coldkeys / IPs (Owner, Axons)"),
                       ("has_axons", "Axons"), ("has_stake", "TotalHotkeyAlpha")]:
        if not df[col].all():
            issues.append(f"{label} missing for {int((~df[col]).sum())} subnets; dependent metrics fall back or are NaN")

    if df["commit_reveal_on"].any():
        nets = df.loc[df["commit_reveal_on"], "netuid"].tolist()
        issues.append(f"Commit-reveal weights on for {nets}: Weights holds the last *revealed* weights, "
                      "which lag the validators' current scoring by the reveal period")
    blocks = meta.get("blocks", [])
    issues.append(f"Blocks in file: {len(blocks)} (block_collection returns block_amount + 1)")
    order = meta.get("netuid_order", [])
    if order:
        # get_subnets_by sorted by summed alpha emission; compare with what the data says
        ranks = [order.index(n) for n in df["netuid"] if n in order]
        alpha = [last_alpha[n] for n in df["netuid"] if n in order]
        r = stats.spearmanr(ranks, alpha).statistic if len(ranks) > 4 else np.nan
        issues.append(f"Spearman(stored netuid order, last-snapshot alpha emission) = {r:.2f} "
                      "(expect strongly negative if the selection ranking matched; alpha is not TAO, "
                      "so selection also favored subnets with cheap alpha)")
        if any(np.isfinite(list(last_tao.values()))) and sum(last_tao.values()) > 0:
            tao = [last_tao[n] for n in df["netuid"] if n in order]
            r2 = stats.spearmanr(ranks, tao).statistic
            issues.append(f"Spearman(stored netuid order, last-snapshot TAO emission) = {r2:.2f}")
    issues.append("Selection bias: subnets were chosen by emission at the final block, favoring subnets healthy now.")
    lay = pd.DataFrame([{"field": k, "layouts_seen": str(v)} for k, v in meta.get("layouts", {}).items()])
    return {"checks_subnets": df, "checks_issues": pd.DataFrame({"issue": issues}), "checks_layouts": lay}
