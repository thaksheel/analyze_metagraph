"""Data checks to read before trusting any result."""

import numpy as np
import pandas as pd
from scipy import stats

from .base import Analysis


class DataChecks(Analysis):
    key = "checks"
    title = "Data checks"
    questions = {
        "Q0": "Is the collected data complete and consistent enough to analyze?"
    }
    tables = {
        "subnets": (
            "Q0",
            "per subnet: snapshots, spacing, uid changes, weight rows vs permits, fields present",
        ),
        "issues": ("Q0", "things to review, in plain language"),
        "layouts": ("Q0", "how each field was parsed (count of snapshots per layout)"),
    }

    def per_subnet(self, sn, panel, ctx):
        permits = (
            int(np.nanmax(np.nansum(np.nan_to_num(sn.vpermit), axis=1))) if sn.T else 0
        )
        return {
            "subnets": pd.DataFrame(
                [
                    {
                        "netuid": sn.netuid,
                        "snapshots": sn.T,
                        "median_spacing_blocks": sn.spacing_blocks,
                        "max_gap_blocks": (
                            int(np.max(np.diff(sn.blocks))) if sn.T > 1 else 0
                        ),
                        "uid_count_changes": int((np.diff(sn.n_uids) != 0).sum()),
                        "max_uids": sn.N,
                        "max_validator_permits": permits,
                        "weight_rows_seen": len(sn.weights_rows),
                        "weights_look_complete": (
                            bool(len(sn.weights_rows) >= 0.5 * permits)
                            if permits
                            else False
                        ),
                        "mechanism_count_max": (
                            float(np.nanmax(sn.mechanism_count))
                            if np.isfinite(sn.mechanism_count).any()
                            else np.nan
                        ),
                        "zero_emission_snapshots": int(
                            (np.nansum(sn.emission, axis=1) == 0).sum()
                        ),
                        "has_hotkeys": sn.has_hotkeys,
                        "has_reg_block": sn.has_reg_block,
                        "has_price": sn.has_price,
                        "has_stake_weight": sn.has_stake_weight,
                        "has_owner_hotkey": bool((sn.owner_hotkey != "").any()),
                        "has_yc3": bool(np.isfinite(sn.yc3).any()),
                        "commit_reveal_on": (
                            bool(np.nanmax(sn.commit_reveal) > 0)
                            if np.isfinite(sn.commit_reveal).any()
                            else False
                        ),
                        "last_subnet_tao": (
                            float(sn.subnet_tao[-1])
                            if np.isfinite(sn.subnet_tao[-1])
                            else np.nan
                        ),
                    }
                ]
            )
        }

    def aggregate(self, results, panel):
        df = self.combined("subnets")
        meta = results.meta
        issues = []
        if (df["mechanism_count_max"] > 1).any():
            issues.append(
                "Subnets with >1 mechanism (only mechanism 0 weights are analyzed): "
                f"{df.loc[df['mechanism_count_max'] > 1, 'netuid'].tolist()}"
            )
        if (~df["weights_look_complete"]).any():
            issues.append(
                "Far fewer weight rows than validator permits: "
                f"{df.loc[~df['weights_look_complete'], 'netuid'].tolist()}"
            )
        if df["commit_reveal_on"].any():
            issues.append(
                f"Commit-reveal on for {df.loc[df['commit_reveal_on'], 'netuid'].tolist()}: Weights are the "
                "last revealed weights and lag current scoring"
            )
        for col, label in [
            ("has_hotkeys", "Keys"),
            ("has_reg_block", "BlockAtRegistration"),
            ("has_price", "SubnetTAO/SubnetAlphaIn"),
            ("has_stake_weight", "StakeWeight"),
            ("has_owner_hotkey", "SubnetOwnerHotkey"),
            ("has_yc3", "Yuma3On"),
        ]:
            if not df[col].all():
                issues.append(f"{label} missing for {int((~df[col]).sum())} subnets")
        order = meta.get("netuid_order", [])
        if order and df["last_subnet_tao"].notna().sum() > 4:
            ranks = [order.index(n) for n in df["netuid"] if n in order]
            tao = [
                df.set_index("netuid").loc[n, "last_subnet_tao"]
                for n in df["netuid"]
                if n in order
            ]
            issues.append(
                f"Spearman(collection order, SubnetTAO at last block) = {stats.spearmanr(ranks, tao).statistic:.2f} "
                "(about -1 if subnets were ranked by SubnetTAO at the last block)"
            )
        issues.append(
            f"{len(meta.get('blocks', []))} blocks; median spacing {df['median_spacing_blocks'].median():.0f} blocks"
        )
        issues.append(
            "Not in the default collection: Bonds, coldkeys, IPs/axons, certificates, per-hotkey stake. "
            "Sybil-like clusters are therefore checked with co-registration, not shared coldkeys/IPs."
        )
        issues.append(
            "Selection bias: subnets were chosen at the last block, favoring subnets healthy now."
        )
        return {
            "issues": pd.DataFrame({"issue": issues}),
            "layouts": pd.DataFrame(
                [
                    {"field": k, "layouts_seen": str(v)}
                    for k, v in meta.get("layouts", {}).items()
                ]
            ),
        }
