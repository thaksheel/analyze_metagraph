"""Sybil-like clusters, found with dynamic multi-layer community detection.

Each window (7 days, sliding 1 day) gets four similarity layers between miners:
  trajectory         correlated log emission share (common shocks removed)
  emission_change    correlated snapshot-to-snapshot changes
  synchrony          coinciding events: emission jumps, shifts in which validators
                     support them, registrations (chance-corrected; network-wide shocks dropped)
  validator_support  overlapping sets of validators that favor them + correlated preference profiles
Miners are linked only when >= 2 layers agree; Louvain finds communities; communities
must be dense and beat random same-size groups; they are tracked across windows, and a
persistent, dense, significant, multi-layer track is "sybil-like".

Coldkeys / IPs are not in the default collection, so identity checks are replaced by
co-registration evidence (BlockAtRegistration) and the synthetic ground-truth test.
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from ..core.stats import corr_matrix, partial_spearman
from .base import Analysis

EPS = 1e-6
LAYERS = ["trajectory", "emission_change", "synchrony", "validator_support"]


def _chance_correct(M: np.ndarray) -> np.ndarray:
    n = len(M)
    if n < 3:
        return M
    m = float(np.nanmean(M[~np.eye(n, dtype=bool)]))
    return np.zeros_like(M) if m >= 1 else np.clip((M - m) / (1 - m), 0, 1)


class SybilAnalysis(Analysis):
    key = "sybil"
    title = "Sybil attack vulnerability"
    problem = "sybil attack vulnerability"
    hypothesis = "15-25% of miners sybil-like in some subnets"
    questions = {
        "Q0": "Can we detect sybil-like clusters (which signals carry the evidence)?",
        "Q1": "What share of miners in each subnet belongs to sybil-like communities?",
        "Q2": "How much emission do they capture relative to their size?",
        "Q3": "Are sybil-heavy subnets less stable, and does YC3 change anything?",
    }
    tables = {
        "windows": (
            "Q1",
            "per subnet and window: miners, candidates, communities found",
        ),
        "communities": (
            "Q0",
            "one row per community per window: density, significance, layers, capture",
        ),
        "members": ("Q1", "one row per community member per window"),
        "tracks": (
            "Q0-Q2",
            "communities followed across windows: persistence, evidence, capture, co-registration",
        ),
        "subnets": (
            "Q1",
            "per subnet: sybil-like tracks, member share, excess emission",
        ),
        "summary": ("Q0-Q2", "detection and size across subnets"),
        "layer_evidence": (
            "Q0",
            "share of pairs on which each layer agrees, sybil-like vs other communities",
        ),
        "stability": (
            "Q3",
            "cross-subnet correlation of member share with rank volatility / disagreement / churn",
        ),
        "yc3": ("Q3", "sybil-like capture with Yuma3 on vs off"),
    }

    # ---------------------------------------------------------------- windows / candidates
    def _windows(self, sn) -> Tuple[List[Tuple[int, int]], int]:
        c = self.cfg.sybil
        W, S = sn.snaps(c.window_days), sn.snaps(c.step_days)
        if sn.T < W:
            return ([(0, sn.T)] if sn.T >= 10 else []), S
        return [(a, a + W) for a in range(0, sn.T - W + 1, S)], S

    def _candidates(self, sn, a: int, b: int):
        c, g = self.cfg.sybil, self.cfg.general
        last = sn.occupant[b - 1]
        present = (
            (sn.occupant[a:b] == last[None, :])
            & sn.is_miner[a:b]
            & (last != "")[None, :]
        )
        share = np.where(present, sn.miner_emi_share[a:b], 0.0)
        mean_share = share.sum(axis=0) / np.maximum(present.sum(axis=0), 1)
        n_m = max(int(sn.is_miner[b - 1].sum()), 1)
        cand = (present.mean(axis=0) >= c.min_presence) & (
            mean_share >= g.min_miner_share_factor / n_m
        )
        return np.where(cand)[0], present, mean_share, n_m

    # ---------------------------------------------------------------- similarity layers
    def layers(
        self, sn, a: int, b: int, uids: np.ndarray, present: np.ndarray
    ) -> Dict[str, np.ndarray]:
        c = self.cfg.sybil
        P = present[:, uids]
        ls = np.log(sn.miner_emi_share[a:b] + EPS)
        miners = sn.is_miner[a:b]
        med = np.array(
            [
                np.median(ls[t][miners[t]]) if miners[t].any() else 0.0
                for t in range(b - a)
            ]
        )
        X = np.where(P, ls[:, uids] - med[:, None], np.nan)
        X = np.where(P, X, np.nanmean(X, axis=0)[None, :])
        out = {"trajectory": corr_matrix(X.T)}
        D = np.diff(X, axis=0)
        both = P[1:] & P[:-1]
        D = np.where(both, D, 0.0)
        out["emission_change"] = corr_matrix(D.T)

        mad = np.array(
            [
                np.median(np.abs(d[m] - np.median(d[m]))) * 1.4826 if m.any() else 0.0
                for d, m in zip(D.T, both.T)
            ]
        )
        E = np.abs(D) >= c.event_z * np.maximum(mad, 1e-3)[None, :]
        if len(sn.weights_rows):
            Wv = sn.W[a:b][:, :, uids]
            num = (Wv[1:] * Wv[:-1]).sum(axis=1)
            den = np.linalg.norm(Wv[1:], axis=1) * np.linalg.norm(Wv[:-1], axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                cos = np.where(den > 0, num / den, 1.0)
            E |= ((1 - cos) >= c.support_change) & both
        coreg = np.zeros((len(uids), len(uids)), bool)
        if sn.has_reg_block:
            rb = sn.reg_block[b - 1, uids]
            inside = np.isfinite(rb) & (rb >= sn.blocks[a]) & (rb <= sn.blocks[b - 1])
            idx = np.clip(
                np.searchsorted(sn.blocks[a:b], rb) - 1, 0, max(len(D) - 1, 0)
            )
            for i in np.where(inside)[0]:
                if len(E):
                    E[idx[i], i] = True
            coreg = (
                inside[:, None]
                & inside[None, :]
                & (np.abs(rb[:, None] - rb[None, :]) <= c.coreg_blocks)
            )
        if len(E):
            E[E.mean(axis=1) >= c.common_event_frac] = False
        Ed = E.copy()
        for k in range(1, c.sync_tolerance + 1):
            Ed[k:] |= E[:-k]
            Ed[:-k] |= E[k:]
        matched = E.astype(float).T @ Ed.astype(float)
        nE = E.sum(axis=0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            sync = (matched + matched.T) / (nE[:, None] + nE[None, :])
        ok = nE >= c.min_events
        sync = _chance_correct(
            np.where(ok[:, None] & ok[None, :], np.nan_to_num(sync), 0.0)
        )
        sync[coreg] = 1.0
        out["synchrony"] = sync

        if len(sn.weights_rows):
            Wv = sn.W[a:b][:, :, uids]
            used = Wv.sum(axis=(0, 2)) > 0
            if used.sum() >= 2:
                Wv = Wv[:, used]
                lw = np.log(Wv + EPS)
                pref = corr_matrix((lw - lw.mean(axis=1, keepdims=True)).mean(axis=0).T)
                mw = Wv.mean(axis=0)
                Sset = (
                    (mw / np.maximum(mw.mean(axis=1, keepdims=True), 1e-12)).T >= 1.0
                ).astype(float)
                inter = Sset @ Sset.T
                union = Sset.sum(1)[:, None] + Sset.sum(1)[None, :] - inter
                with np.errstate(invalid="ignore", divide="ignore"):
                    jac = np.where(union > 0, inter / union, 0.0)
                out["validator_support"] = 0.5 * np.clip(
                    pref, 0, 1
                ) + 0.5 * _chance_correct(jac)
        return out

    def combine(self, layers: Dict[str, np.ndarray]):
        c = self.cfg.sybil
        names = [k for k in LAYERS if k in layers and c.layer_weights.get(k, 0) > 0]
        w = np.array([c.layer_weights[k] for k in names])
        stack = np.stack([np.clip(layers[k], 0, 1) for k in names])
        S = (stack * w[:, None, None]).sum(axis=0) / w.sum()
        strong = (stack >= c.layer_strong).sum(axis=0)
        np.fill_diagonal(S, 0)
        np.fill_diagonal(strong, 0)
        return S, strong, names, stack

    # ---------------------------------------------------------------- per subnet
    def per_subnet(self, sn, panel, ctx):
        c = self.cfg.sybil
        rng = np.random.default_rng(self.cfg.seed + sn.netuid)
        wins, step = self._windows(sn)
        m_alpha, m_tao = sn.miner_flow("alpha"), sn.miner_flow("tao")
        comm_rows, mem_rows, win_rows, tracks = [], [], [], []
        for wi, (a, b) in enumerate(wins):
            uids, present, mean_share, n_m = self._candidates(sn, a, b)
            win_rows.append(
                {
                    "netuid": sn.netuid,
                    "window": wi,
                    "t_start": a,
                    "t_end": b,
                    "n_miners": n_m,
                    "n_candidates": len(uids),
                    "yc3": (
                        float(np.nanmean(sn.yc3[a:b]))
                        if np.isfinite(sn.yc3[a:b]).any()
                        else np.nan
                    ),
                }
            )
            comms = []
            if len(uids) >= c.min_size:
                S, strong, names, stack = self.combine(
                    self.layers(sn, a, b, uids, present)
                )
                adj = (S >= c.edge_min) & (strong >= min(c.min_layers, len(names)))
                G = nx.Graph()
                ii, jj = np.where(np.triu(adj, 1))
                G.add_weighted_edges_from(
                    (int(i), int(j), float(S[i, j])) for i, j in zip(ii, jj)
                )
                parts = (
                    nx.community.louvain_communities(
                        G, weight="weight", resolution=c.resolution, seed=self.cfg.seed
                    )
                    if G.number_of_edges()
                    else []
                )
                vol_all = S.sum()
                for part in parts:
                    idx = np.array(sorted(part))
                    k = len(idx)
                    if k < c.min_size:
                        continue
                    sub, iu = np.ix_(idx, idx), np.triu_indices(k, 1)
                    internal = float(S[sub][iu].mean())
                    vol = S[idx].sum()
                    null = np.array(
                        [
                            S[np.ix_(r, r)][iu].mean()
                            for r in (
                                rng.choice(len(uids), k, replace=False)
                                for _ in range(c.null_draws)
                            )
                        ]
                    )
                    members = uids[idx]
                    emi, uid_sh = float(mean_share[members].sum()), k / n_m
                    excess = max(emi - uid_sh, 0.0)
                    com = {
                        "netuid": sn.netuid,
                        "window": wi,
                        "t_start": a,
                        "t_end": b,
                        "size": k,
                        "density": float(adj[sub][iu].mean()),
                        "internal_similarity": internal,
                        "conductance": float(
                            (vol - S[sub].sum()) / max(min(vol, vol_all - vol), 1e-12)
                        ),
                        "null_p": float(
                            (1 + (null >= internal).sum()) / (1 + len(null))
                        ),
                        "null_z": float(
                            (internal - null.mean()) / (null.std() or 1e-9)
                        ),
                        "mean_strong_layers": float(strong[sub][iu].mean()),
                        "uid_share": uid_sh,
                        "emission_share": emi,
                        "capture_ratio": emi / uid_sh if uid_sh else np.nan,
                        "excess_share": excess,
                        # only the last `step` snapshots, so summing over windows counts each once
                        "excess_alpha_step": excess
                        * m_alpha[max(a, b - step) : b].sum(),
                        "excess_tao_step": excess * m_tao[max(a, b - step) : b].sum(),
                        "yc3": win_rows[-1]["yc3"],
                    }
                    com["qualified"] = bool(
                        com["density"] >= c.min_density and com["null_p"] <= c.null_p
                    )
                    for li, name in enumerate(names):
                        com[f"layer_{name}"] = float(stack[li][sub][iu].mean())
                        com[f"strong_{name}"] = float(
                            (stack[li][sub][iu] >= c.layer_strong).mean()
                        )
                    com["_occ"] = {sn.occupant[b - 1, u] for u in members}
                    com["_uids"] = members.tolist()
                    comms.append(com)
            win_rows[-1]["n_communities"] = len(comms)
            # track across windows (greedy best overlap; a track may skip one window)
            open_tr = [tr for tr in tracks if tr["last_window"] >= wi - 2]
            pairs = sorted(
                (
                    (
                        len(cm["_occ"] & tr["members"])
                        / len(cm["_occ"] | tr["members"]),
                        ci,
                        tr["id"],
                    )
                    for ci, cm in enumerate(comms)
                    for tr in open_tr
                ),
                reverse=True,
            )
            assign, used_c, used_t = {}, set(), set()
            for jac, ci, tid in pairs:
                if jac >= c.match_jaccard and ci not in used_c and tid not in used_t:
                    assign[ci] = tid
                    used_c.add(ci)
                    used_t.add(tid)
            for ci, cm in enumerate(comms):
                tr = (
                    next(t for t in tracks if t["id"] == assign[ci])
                    if ci in assign
                    else None
                )
                if tr is None:
                    tr = {"id": f"{sn.netuid}-T{len(tracks)}", "counts": {}}
                    tracks.append(tr)
                tr["members"], tr["last_window"] = cm["_occ"], wi
                for o in cm["_occ"]:
                    tr["counts"][o] = tr["counts"].get(o, 0) + 1
                cm["track_id"] = tr["id"]
                mem_rows += [
                    {
                        "netuid": sn.netuid,
                        "window": wi,
                        "t_end": b,
                        "track_id": tr["id"],
                        "uid": int(u),
                        "occupant": sn.occupant[b - 1, u],
                    }
                    for u in cm["_uids"]
                ]
                comm_rows.append({k: v for k, v in cm.items() if not k.startswith("_")})

        cdf, mdf = pd.DataFrame(comm_rows), pd.DataFrame(mem_rows)
        reg_of = {}
        if sn.has_reg_block:
            for t in range(sn.T):
                for u in range(sn.N):
                    if np.isfinite(sn.reg_block[t, u]) and sn.occupant[t, u]:
                        reg_of[sn.occupant[t, u]] = sn.reg_block[t, u]
        track_rows = []
        for tid, g in (cdf.groupby("track_id") if len(cdf) else []):
            tr = next(t for t in tracks if t["id"] == tid)
            n_w = len(g)
            core = [o for o, k in tr["counts"].items() if k >= 0.5 * n_w]
            rec = {
                "netuid": sn.netuid,
                "track_id": tid,
                "n_windows": n_w,
                "first_window": int(g["window"].min()),
                "last_window": int(g["window"].max()),
                "span_days": float(
                    (
                        sn.hours[int(g["t_end"].max()) - 1]
                        - sn.hours[int(g["t_start"].min())]
                    )
                    / 24
                ),
                "mean_size": float(g["size"].mean()),
                "max_size": int(g["size"].max()),
                "core_size": len(core),
                "mean_density": float(g["density"].mean()),
                "median_null_p": float(g["null_p"].median()),
                "mean_null_z": float(g["null_z"].mean()),
                "mean_strong_layers": float(g["mean_strong_layers"].mean()),
                "mean_conductance": float(g["conductance"].mean()),
                "mean_capture_ratio": float(g["capture_ratio"].mean()),
                "excess_alpha": float(g["excess_alpha_step"].sum()),
                "excess_tao": float(g["excess_tao_step"].sum()),
            }
            for name in LAYERS:
                if f"strong_{name}" in g:
                    rec[f"strong_{name}"] = float(g[f"strong_{name}"].mean())
            rb = np.array([reg_of[o] for o in core if o in reg_of])
            rec["core_coregistered_share"] = (
                float(
                    np.mean([(np.abs(rb - x) <= c.coreg_blocks).sum() > 1 for x in rb])
                )
                if len(rb) >= 2
                else np.nan
            )
            rec["core_registration_span_blocks"] = (
                float(np.ptp(rb)) if len(rb) else np.nan
            )
            rec["sybil_like"] = bool(
                n_w >= c.min_track_windows
                and rec["median_null_p"] <= c.null_p
                and rec["mean_density"] >= c.min_density
                and rec["mean_strong_layers"] >= c.min_layers
            )
            rec["extractive"] = bool(
                rec["sybil_like"] and rec["mean_capture_ratio"] > 1.0
            )
            track_rows.append(rec)
        tdf = pd.DataFrame(track_rows)
        syb = set(tdf.loc[tdf["sybil_like"], "track_id"]) if len(tdf) else set()
        if len(cdf):
            cdf["sybil_like_track"] = cdf["track_id"].isin(syb)
            mdf["sybil_like_track"] = mdf["track_id"].isin(syb)
        wdf = pd.DataFrame(win_rows)
        per_w = (
            mdf[mdf["sybil_like_track"]].groupby("window")["occupant"].nunique()
            if len(mdf)
            else pd.Series(dtype=float)
        )
        share_w = (
            per_w.reindex(wdf["window"]).fillna(0).to_numpy()
            / wdf["n_miners"].clip(lower=1).to_numpy()
            if len(wdf)
            else np.array([])
        )
        in_syb = (
            set(mdf.loc[mdf["sybil_like_track"], "occupant"]) if len(mdf) else set()
        )
        non = sn.is_miner & ~np.isin(sn.occupant, list(in_syb))
        summ = {
            "netuid": sn.netuid,
            "windows": len(wdf),
            "communities": len(cdf),
            "qualified_communities": int(cdf["qualified"].sum()) if len(cdf) else 0,
            "tracks": len(tdf),
            "sybil_like_tracks": len(syb),
            "extractive_tracks": int(tdf["extractive"].sum()) if len(tdf) else 0,
            "sybil_member_share_mean": (
                float(share_w.mean()) if len(share_w) else np.nan
            ),
            "sybil_member_share_max": float(share_w.max()) if len(share_w) else np.nan,
            "sybil_excess_alpha": (
                float(tdf.loc[tdf["sybil_like"], "excess_alpha"].sum())
                if len(tdf)
                else 0.0
            ),
            "sybil_excess_tao": (
                float(tdf.loc[tdf["sybil_like"], "excess_tao"].sum())
                if len(tdf)
                else 0.0
            ),
            "non_member_turnover": (
                float(sn.replaced[non].mean()) if non.any() else np.nan
            ),
            "subnet_alpha_total": float(sn.alpha_flow.sum()),
            "yc3_on_share": (
                float(np.nanmean(sn.yc3)) if np.isfinite(sn.yc3).any() else np.nan
            ),
        }
        return {
            "windows": wdf,
            "communities": cdf,
            "members": mdf,
            "tracks": tdf,
            "subnets": pd.DataFrame([summ]),
        }

    # ---------------------------------------------------------------- aggregate
    def aggregate(self, results, panel):
        c = self.cfg.sybil
        s, t, cm = (
            self.combined("subnets"),
            self.combined("tracks"),
            self.combined("communities"),
        )
        out = {}
        if not len(s):
            return out
        syb = t[t["sybil_like"]] if len(t) else t
        out["summary"] = pd.DataFrame(
            [
                {
                    "subnets": len(s),
                    "share_subnets_with_sybil_like_track": float(
                        (s["sybil_like_tracks"] > 0).mean()
                    ),
                    "share_subnets_member_share_above_threshold": float(
                        (s["sybil_member_share_max"] >= c.share_threshold).mean()
                    ),
                    "median_member_share": float(s["sybil_member_share_mean"].median()),
                    "p90_member_share": float(
                        s["sybil_member_share_max"].quantile(0.9)
                    ),
                    "communities": len(cm),
                    "share_communities_qualified": (
                        float(cm["qualified"].mean()) if len(cm) else np.nan
                    ),
                    "tracks": len(t),
                    "sybil_like_tracks": len(syb),
                    "share_sybil_tracks_extractive": (
                        float(syb["extractive"].mean()) if len(syb) else np.nan
                    ),
                    "median_capture_ratio_sybil_tracks": (
                        float(syb["mean_capture_ratio"].median())
                        if len(syb)
                        else np.nan
                    ),
                    "share_sybil_tracks_coregistered": (
                        float((syb["core_coregistered_share"] >= 0.5).mean())
                        if len(syb)
                        else np.nan
                    ),
                    "median_sybil_track_span_days": (
                        float(syb["span_days"].median()) if len(syb) else np.nan
                    ),
                    "excess_share_of_emission": float(
                        (
                            s["sybil_excess_alpha"]
                            / s["subnet_alpha_total"].replace(0, np.nan)
                        ).mean()
                    ),
                }
            ]
        )
        out["layer_evidence"] = (
            pd.DataFrame(
                [
                    {
                        "layer": name,
                        "agreeing_pairs_sybil_like": (
                            float(syb[f"strong_{name}"].mean())
                            if len(syb) and f"strong_{name}" in syb
                            else np.nan
                        ),
                        "agreeing_pairs_other": (
                            float(t.loc[~t["sybil_like"], f"strong_{name}"].mean())
                            if f"strong_{name}" in t
                            else np.nan
                        ),
                    }
                    for name in LAYERS
                ]
            )
            if len(t)
            else pd.DataFrame()
        )
        # Q3: stability vs member share across subnets
        stab = (
            panel.groupby("netuid")
            .agg(
                rank_corr=("rank_corr", "mean"),
                agreement=("validator_agreement", "mean"),
                n_uids=("n_uids", "mean"),
            )
            .reset_index()
        )
        m = s.merge(stab, on="netuid")
        m["rank_volatility"], m["validator_disagreement"] = (
            1 - m["rank_corr"],
            1 - m["agreement"],
        )
        rows = []
        for y in ("rank_volatility", "validator_disagreement", "non_member_turnover"):
            x, yy = m["sybil_member_share_mean"].to_numpy(float), m[y].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(yy)
            r = (
                stats.spearmanr(x[ok], yy[ok])
                if ok.sum() >= 5 and np.ptp(x[ok]) > 0 and np.ptp(yy[ok]) > 0
                else None
            )
            pr, pp = partial_spearman(x, yy, m["n_uids"].to_numpy(float))
            rows.append(
                {
                    "outcome": y,
                    "n": int(ok.sum()),
                    "spearman_r": float(r.statistic) if r else np.nan,
                    "p_value": float(r.pvalue) if r else np.nan,
                    "partial_r_ctrl_size": pr,
                    "partial_p": pp,
                }
            )
        out["stability"] = pd.DataFrame(rows)
        yc = []
        if len(cm) and cm["yc3"].notna().any():
            q = cm[cm["sybil_like_track"]] if "sybil_like_track" in cm else cm
            on, off = (
                q.loc[q["yc3"] >= 0.5, "capture_ratio"].dropna(),
                q.loc[q["yc3"] < 0.5, "capture_ratio"].dropna(),
            )
            if len(on) >= 3 and len(off) >= 3:
                u = stats.mannwhitneyu(on, off)
                yc.append(
                    {
                        "comparison": "sybil-like communities, YC3 on vs off",
                        "n_on": len(on),
                        "n_off": len(off),
                        "median_ratio_on": on.median(),
                        "median_ratio_off": off.median(),
                        "p_value": u.pvalue,
                    }
                )
        out["yc3"] = pd.DataFrame(yc)
        return out

    def rank_entry(self, results) -> Dict:
        s = results.get("sybil_subnets").set_index("netuid")
        summ = results.get("sybil_summary").iloc[0]
        return {
            "size_metric": f"share of subnets with >= {self.cfg.sybil.share_threshold:.0%} miners in sybil-like communities",
            "size_value": summ["share_subnets_member_share_above_threshold"],
            "affected_alpha": s["sybil_excess_alpha"],
            "affected_tao": s["sybil_excess_tao"],
            "measured": (
                f"sybil-like tracks in {summ['share_subnets_with_sybil_like_track']:.0%} of subnets; "
                f"p90 member share {summ['p90_member_share']:.1%}; "
                f"{summ['share_sybil_tracks_coregistered']:.0%} of tracks co-registered"
                if np.isfinite(summ["share_sybil_tracks_coregistered"])
                else f"sybil-like tracks in {summ['share_subnets_with_sybil_like_track']:.0%} of subnets"
            ),
        }
