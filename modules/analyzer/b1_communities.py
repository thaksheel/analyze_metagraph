"""B1 — Can we detect sybil-like clusters?

Dynamic, multi-layer community detection:

1. Slide a window (default 7 days, 1-day step) over each subnet.
2. In each window, build four similarity layers between miners:
     trajectory        correlation of log emission share (common-mode removed)
     emission_change   correlation of snapshot-to-snapshot changes in log share
     synchrony         how often their events coincide (emission jumps, changes in
                       which validators support them, registrations, serve_axon and
                       commitment updates), chance-corrected,
                       with network-wide shocks removed
     validator_support overlap of the validators that favor them (chance-corrected
                       Jaccard) plus correlation of validator-preference profiles
3. Link two miners only when the combined similarity is high AND at least
   `dc_min_layers` layers agree, so one coincidence is never enough.
4. Run Louvain on the window graph; keep communities that are dense and whose
   internal similarity beats random groups of the same size (permutation test).
5. Track communities across windows by member overlap. A track is sybil-like when
   it is persistent, dense, significant and multi-layer.

Coldkeys / IPs (identity file) are used only afterwards, to measure precision and
recall of what the behavioral method found.
"""
from itertools import combinations
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .adapters import ip_prefix
from .config import Config
from .metrics import corr_matrix
from .subnet import SubnetData

EPS = 1e-6
LAYERS = ["trajectory", "emission_change", "synchrony", "validator_support"]


def _windows(sd: SubnetData, cfg: Config) -> Tuple[List[Tuple[int, int]], int]:
    W = sd.snaps(cfg.dc_window_days)
    S = sd.snaps(cfg.dc_step_days)
    if sd.T < W:
        return ([(0, sd.T)] if sd.T >= 10 else []), S
    return [(a, a + W) for a in range(0, sd.T - W + 1, S)], S


def _candidates(sd: SubnetData, a: int, b: int, cfg: Config):
    last = sd.occupant[b - 1]
    present = (sd.occupant[a:b] == last[None, :]) & sd.is_miner[a:b] & (last != "")[None, :]
    frac = present.mean(axis=0)
    share = np.where(present, sd.miner_emi_share[a:b], 0.0)
    mean_share = share.sum(axis=0) / np.maximum(present.sum(axis=0), 1)
    n_m = max(int(sd.is_miner[b - 1].sum()), 1)
    cand = (frac >= cfg.dc_min_presence) & (mean_share >= cfg.min_miner_share_factor / n_m)
    return np.where(cand)[0], present, mean_share, n_m


def _chance_correct(M: np.ndarray) -> np.ndarray:
    n = len(M)
    if n < 3:
        return M
    off = M[~np.eye(n, dtype=bool)]
    m = float(np.nanmean(off))
    if m >= 1:
        return np.zeros_like(M)
    return np.clip((M - m) / (1 - m), 0, 1)


def _layers(sd: SubnetData, a: int, b: int, uids: np.ndarray, present: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    P = present[:, uids]
    ls = np.log(sd.miner_emi_share[a:b] + EPS)
    miners = sd.is_miner[a:b]
    med = np.array([np.median(ls[t][miners[t]]) if miners[t].any() else 0.0 for t in range(b - a)])
    X = np.where(P, ls[:, uids] - med[:, None], np.nan)
    col_mean = np.nanmean(np.where(P, X, np.nan), axis=0)
    X = np.where(P, X, col_mean[None, :])
    out = {"trajectory": corr_matrix(X.T)}

    D = np.diff(X, axis=0)
    both = P[1:] & P[:-1]
    D = np.where(both, D, 0.0)
    out["emission_change"] = corr_matrix(D.T)

    # events: emission jumps, changes in who supports the miner, registration
    mad = np.array([np.median(np.abs(d[m] - np.median(d[m]))) * 1.4826 if m.any() else 0.0
                    for d, m in zip(D.T, both.T)])
    E = np.abs(D) >= cfg.dc_event_z * np.maximum(mad, 1e-3)[None, :]
    if len(sd.weights_rows):
        Wv = sd.W[a:b][:, :, uids]                                     # [w, K, n]
        num = (Wv[1:] * Wv[:-1]).sum(axis=1)
        den = np.linalg.norm(Wv[1:], axis=1) * np.linalg.norm(Wv[:-1], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(den > 0, num / den, 1.0)
        E |= ((1 - cos) >= cfg.dc_support_change) & both
    # synchronized serve_axon / commitment updates (miner-emitted events)
    for arr in (sd.axon_block, sd.commit_block):
        A = arr[a:b][:, uids]
        if np.isfinite(A).any() and len(E):
            with np.errstate(invalid="ignore"):
                E |= (np.nan_to_num(A[1:]) > np.nan_to_num(A[:-1])) & both
    coreg = np.zeros((len(uids), len(uids)), bool)
    if sd.has_reg_block:
        rb = sd.reg_block[b - 1, uids]
        inside = np.isfinite(rb) & (rb >= sd.blocks[a]) & (rb <= sd.blocks[b - 1])
        idx = np.clip(np.searchsorted(sd.blocks[a:b], rb) - 1, 0, max(len(D) - 1, 0))
        for i in np.where(inside)[0]:
            if len(E):
                E[idx[i], i] = True
        coreg = inside[:, None] & inside[None, :] & (np.abs(rb[:, None] - rb[None, :]) <= cfg.coreg_blocks)
    if len(E):
        E[E.mean(axis=1) >= cfg.dc_common_event_frac] = False      # network-wide shocks
    Ed = E.copy()
    for k in range(1, cfg.dc_sync_tolerance + 1):
        Ed[k:] |= E[:-k]
        Ed[:-k] |= E[k:]
    Ef, Edf = E.astype(float), Ed.astype(float)
    matched = Ef.T @ Edf                                               # i's events matched by j
    nE = Ef.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sync = (matched + matched.T) / (nE[:, None] + nE[None, :])
    ok = (nE >= cfg.dc_min_events)
    sync = np.where(ok[:, None] & ok[None, :], np.nan_to_num(sync), 0.0)
    sync = _chance_correct(sync)
    sync[coreg] = 1.0
    out["synchrony"] = sync

    if len(sd.weights_rows):
        Wv = sd.W[a:b][:, :, uids]
        used = Wv.sum(axis=(0, 2)) > 0
        if used.sum() >= 2:
            Wv = Wv[:, used]
            lw = np.log(Wv + EPS)
            pref = corr_matrix((lw - lw.mean(axis=1, keepdims=True)).mean(axis=0).T)
            mw = Wv.mean(axis=0)                                       # [K, n]
            rel = mw / np.maximum(mw.mean(axis=1, keepdims=True), 1e-12)
            S = (rel.T >= 1.0).astype(float)                           # [n, K] support sets
            inter = S @ S.T
            union = S.sum(1)[:, None] + S.sum(1)[None, :] - inter
            with np.errstate(invalid="ignore", divide="ignore"):
                jac = np.where(union > 0, inter / union, 0.0)
            out["validator_support"] = 0.5 * np.clip(pref, 0, 1) + 0.5 * _chance_correct(jac)
    return out


def _combine(layers: Dict[str, np.ndarray], cfg: Config):
    names = [k for k in LAYERS if k in layers and cfg.dc_layer_weights.get(k, 0) > 0]
    w = np.array([cfg.dc_layer_weights[k] for k in names])
    stack = np.stack([np.clip(layers[k], 0, 1) for k in names])
    S = (stack * w[:, None, None]).sum(axis=0) / w.sum()
    strong = (stack >= cfg.dc_layer_strong).sum(axis=0)
    np.fill_diagonal(S, 0)
    np.fill_diagonal(strong, 0)
    return S, strong, names, stack


def per_subnet(sd: SubnetData, panel: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed + sd.netuid)
    wins, step = _windows(sd, cfg)
    miner_alpha = np.where(sd.is_miner, sd.alpha_flow, 0).sum(axis=1)
    miner_tao = np.where(sd.is_miner, sd.tao_flow, 0).sum(axis=1)
    comm_rows, mem_rows, win_rows = [], [], []
    tracks: List[dict] = []

    for wi, (a, b) in enumerate(wins):
        uids, present, mean_share, n_m = _candidates(sd, a, b, cfg)
        wrec = {"netuid": sd.netuid, "window": wi, "t_start": a, "t_end": b,
                "n_miners": n_m, "n_candidates": len(uids)}
        comms = []
        if len(uids) >= cfg.dc_min_size:
            layers = _layers(sd, a, b, uids, present, cfg)
            S, strong, names, stack = _combine(layers, cfg)
            adj = (S >= cfg.dc_edge_min) & (strong >= min(cfg.dc_min_layers, len(names)))
            G = nx.Graph()
            ii, jj = np.where(np.triu(adj, 1))
            G.add_weighted_edges_from((int(i), int(j), float(S[i, j])) for i, j in zip(ii, jj))
            parts = nx.community.louvain_communities(G, weight="weight", resolution=cfg.dc_resolution,
                                                     seed=cfg.seed) if G.number_of_edges() else []
            vol_all = S.sum()
            for part in parts:
                idx = np.array(sorted(part))
                k = len(idx)
                if k < cfg.dc_min_size:
                    continue
                sub = np.ix_(idx, idx)
                iu = np.triu_indices(k, 1)
                density = float(adj[sub][iu].mean())
                internal = float(S[sub][iu].mean())
                vol = S[idx].sum()
                cut = vol - S[sub].sum()
                conduct = float(cut / max(min(vol, vol_all - vol), 1e-12))
                null = np.array([S[np.ix_(r, r)][iu].mean() for r in
                                 (rng.choice(len(uids), k, replace=False) for _ in range(cfg.dc_null_draws))])
                p = float((1 + (null >= internal).sum()) / (1 + len(null)))
                z = float((internal - null.mean()) / (null.std() or 1e-9))
                members = uids[idx]
                emi = float(mean_share[members].sum())
                uid_sh = k / n_m
                excess = max(emi - uid_sh, 0.0)
                occ = [sd.occupant[b - 1, u] for u in members]
                c = {"netuid": sd.netuid, "window": wi, "t_start": a, "t_end": b, "size": k,
                     "density": density, "internal_similarity": internal, "conductance": conduct,
                     "null_p": p, "null_z": z, "mean_strong_layers": float(strong[sub][iu].mean()),
                     "uid_share": uid_sh, "emission_share": emi,
                     "capture_ratio": emi / uid_sh if uid_sh else np.nan, "excess_share": excess,
                     # excess over the last `step` snapshots only, so summing windows counts each once
                     "excess_alpha_step": excess * miner_alpha[max(a, b - step):b].sum(),
                     "excess_tao_step": excess * miner_tao[max(a, b - step):b].sum(),
                     "qualified": bool(density >= cfg.dc_min_density and p <= cfg.dc_null_p)}
                for li, name in enumerate(names):
                    c[f"layer_{name}"] = float(stack[li][sub][iu].mean())
                    c[f"strong_{name}"] = float((stack[li][sub][iu] >= cfg.dc_layer_strong).mean())
                c["_occ"] = set(occ)
                c["_uids"] = members.tolist()
                comms.append(c)
        wrec["n_communities"] = len(comms)
        win_rows.append(wrec)

        # ---- track communities across windows ----
        open_tracks = [tr for tr in tracks if tr["last_window"] >= wi - 2]
        pairs = []
        for ci, c in enumerate(comms):
            for tr in open_tracks:
                inter = len(c["_occ"] & tr["members"])
                jac = inter / len(c["_occ"] | tr["members"])
                if jac >= cfg.dc_match_jaccard:
                    pairs.append((jac, ci, tr["id"]))
        used_c, used_t = set(), set()
        assign = {}
        for jac, ci, tid in sorted(pairs, reverse=True):
            if ci in used_c or tid in used_t:
                continue
            assign[ci] = tid
            used_c.add(ci)
            used_t.add(tid)
        for ci, c in enumerate(comms):
            if ci in assign:
                tr = next(t for t in tracks if t["id"] == assign[ci])
            else:
                tr = {"id": f"{sd.netuid}-T{len(tracks)}", "windows": [], "counts": {}}
                tracks.append(tr)
            tr["members"] = c["_occ"]
            tr["last_window"] = wi
            tr["windows"].append(wi)
            for o in c["_occ"]:
                tr["counts"][o] = tr["counts"].get(o, 0) + 1
            c["track_id"] = tr["id"]
            for u, o in zip(c["_uids"], [sd.occupant[b - 1, u] for u in c["_uids"]]):
                mem_rows.append({"netuid": sd.netuid, "window": wi, "t_end": b, "track_id": tr["id"],
                                 "uid": int(u), "occupant": o})
            comm_rows.append({k: v for k, v in c.items() if not k.startswith("_")})

    cdf = pd.DataFrame(comm_rows)
    mdf = pd.DataFrame(mem_rows)
    track_rows = []
    if len(cdf):
        for tid, g in cdf.groupby("track_id"):
            tr = next(t for t in tracks if t["id"] == tid)
            n_w = len(g)
            core = [o for o, c in tr["counts"].items() if c >= 0.5 * n_w]
            rec = {"netuid": sd.netuid, "track_id": tid, "n_windows": n_w,
                   "first_window": int(g["window"].min()), "last_window": int(g["window"].max()),
                   "span_days": float((sd.hours[int(g["t_end"].max()) - 1] - sd.hours[int(g["t_start"].min())]) / 24),
                   "mean_size": float(g["size"].mean()), "max_size": int(g["size"].max()),
                   "core_size": len(core), "mean_density": float(g["density"].mean()),
                   "median_null_p": float(g["null_p"].median()), "mean_null_z": float(g["null_z"].mean()),
                   "mean_strong_layers": float(g["mean_strong_layers"].mean()),
                   "mean_conductance": float(g["conductance"].mean()),
                   "mean_capture_ratio": float(g["capture_ratio"].mean()),
                   "excess_alpha": float(g["excess_alpha_step"].sum()),
                   "excess_tao": float(g["excess_tao_step"].sum())}
            for name in LAYERS:
                if f"strong_{name}" in g:
                    rec[f"strong_{name}"] = float(g[f"strong_{name}"].mean())
            rec["sybil_like"] = bool(n_w >= cfg.dc_min_track_windows and rec["median_null_p"] <= cfg.dc_null_p
                                     and rec["mean_density"] >= cfg.dc_min_density
                                     and rec["mean_strong_layers"] >= cfg.dc_min_layers)
            rec["extractive"] = bool(rec["sybil_like"] and rec["mean_capture_ratio"] > 1.0)
            if sd.identity:
                rec["core_same_coldkey_rate"] = _pair_rate([sd.coldkey(o) for o in core])
                rec["core_same_ip24_rate"] = _pair_rate([ip_prefix(sd.ip(o)) for o in core])
            rec["_core"] = core
            track_rows.append(rec)
    tdf = pd.DataFrame(track_rows)
    sybil_tracks = set(tdf.loc[tdf["sybil_like"], "track_id"]) if len(tdf) else set()
    if len(cdf):
        cdf["sybil_like_track"] = cdf["track_id"].isin(sybil_tracks)
        mdf["sybil_like_track"] = mdf["track_id"].isin(sybil_tracks)

    # subnet summary + identity-based validation
    wdf = pd.DataFrame(win_rows)
    if len(mdf):
        per_w = mdf[mdf["sybil_like_track"]].groupby("window")["occupant"].nunique()
    else:
        per_w = pd.Series(dtype=float)
    share_w = (per_w.reindex(wdf["window"]).fillna(0).to_numpy() / wdf["n_miners"].clip(lower=1).to_numpy()) if len(wdf) else np.array([])
    summ = {"netuid": sd.netuid, "windows": len(wdf), "communities": len(cdf),
            "qualified_communities": int(cdf["qualified"].sum()) if len(cdf) else 0,
            "tracks": len(tdf), "sybil_like_tracks": len(sybil_tracks),
            "extractive_tracks": int(tdf["extractive"].sum()) if len(tdf) else 0,
            "sybil_member_share_mean": float(share_w.mean()) if len(share_w) else np.nan,
            "sybil_member_share_max": float(share_w.max()) if len(share_w) else np.nan,
            "sybil_excess_alpha": float(tdf.loc[tdf["sybil_like"], "excess_alpha"].sum()) if len(tdf) else 0.0,
            "sybil_excess_tao": float(tdf.loc[tdf["sybil_like"], "excess_tao"].sum()) if len(tdf) else 0.0,
            "subnet_alpha_total": float(sd.alpha_flow.sum())}
    if sd.identity and len(wdf):
        # recall: miners in same-coldkey groups of >= dc_min_size found in a sybil-like core
        mid = wins[len(wins) // 2]
        uids, *_ = _candidates(sd, mid[0], mid[1], cfg)
        occ = [sd.occupant[mid[1] - 1, u] for u in uids]
        ck = pd.Series([sd.coldkey(o) for o in occ], index=occ).dropna()
        sizes = ck.map(ck.value_counts())
        grouped = set(sizes[sizes >= cfg.dc_min_size].index)
        found = set().union(*[set(c) for c in tdf.loc[tdf["sybil_like"], "_core"]]) if len(sybil_tracks) else set()
        summ["coldkey_group_miners"] = len(grouped)
        summ["coldkey_group_found"] = len(grouped & found)
        summ["baseline_same_coldkey_rate"] = _pair_rate(list(ck.values))
    if len(tdf):
        tdf = tdf.drop(columns=["_core"])
    return {"b1_windows": wdf, "b1_communities": cdf, "b1_members": mdf, "b1_tracks": tdf,
            "b1_subnets": pd.DataFrame([summ])}


def _pair_rate(keys) -> float:
    keys = [k for k in keys if k]
    if len(keys) < 2:
        return np.nan
    return float(np.mean([x == y for x, y in combinations(keys, 2)]))


def aggregate(R: Dict[str, pd.DataFrame], cfg: Config) -> Dict[str, pd.DataFrame]:
    s = R.get("b1_subnets", pd.DataFrame())
    t = R.get("b1_tracks", pd.DataFrame())
    c = R.get("b1_communities", pd.DataFrame())
    if not len(s):
        return {}
    syb = t[t["sybil_like"]] if len(t) else t
    summ = {
        "subnets": len(s),
        "share_subnets_with_sybil_like_track": float((s["sybil_like_tracks"] > 0).mean()),
        "share_subnets_member_share_above_threshold": float((s["sybil_member_share_max"] >= cfg.sybil_share_threshold).mean()),
        "median_sybil_member_share": float(s["sybil_member_share_mean"].median()),
        "p90_sybil_member_share": float(s["sybil_member_share_max"].quantile(0.9)),
        "communities_found": len(c),
        "share_communities_qualified": float(c["qualified"].mean()) if len(c) else np.nan,
        "tracks": len(t), "sybil_like_tracks": len(syb),
        "share_sybil_tracks_extractive": float(syb["extractive"].mean()) if len(syb) else np.nan,
        "median_sybil_track_span_days": float(syb["span_days"].median()) if len(syb) else np.nan,
        "sybil_excess_share_of_emission": float((s["sybil_excess_alpha"] / s["subnet_alpha_total"].replace(0, np.nan)).mean()),
    }
    if len(syb) and "core_same_coldkey_rate" in syb:
        hit = (syb["core_same_coldkey_rate"] >= 0.5) | (syb.get("core_same_ip24_rate", 0) >= 0.5)
        summ["precision_identity_confirmed"] = float(hit.mean())
        summ["core_same_coldkey_rate"] = float(syb["core_same_coldkey_rate"].mean())
        summ["baseline_same_coldkey_rate"] = float(s["baseline_same_coldkey_rate"].mean())
        if "coldkey_group_miners" in s and s["coldkey_group_miners"].sum() > 0:
            summ["recall_coldkey_groups"] = float(s["coldkey_group_found"].sum() / s["coldkey_group_miners"].sum())
    # agreement with the static Area 5 clusters
    m1, m5 = R.get("b1_members", pd.DataFrame()), R.get("a5_members", pd.DataFrame())
    if len(m1) and len(m5):
        a = set(zip(m1.loc[m1["sybil_like_track"], "netuid"], m1.loc[m1["sybil_like_track"], "occupant"]))
        b = set(zip(m5["netuid"], m5["occupant"]))
        summ["share_a5_members_in_b1"] = len(a & b) / len(b) if b else np.nan
        summ["share_b1_members_in_a5"] = len(a & b) / len(a) if a else np.nan
    out = {"b1_summary": pd.DataFrame([summ])}

    # which layers carry the signal inside sybil-like tracks vs other communities
    rows = []
    if len(t):
        for name in LAYERS:
            col = f"strong_{name}"
            if col in t:
                rows.append({"layer": name,
                             "agreeing_pairs_sybil_like": float(syb[col].mean()) if len(syb) else np.nan,
                             "agreeing_pairs_other": float(t.loc[~t["sybil_like"], col].mean())})
    out["b1_layer_evidence"] = pd.DataFrame(rows)
    return out
