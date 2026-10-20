"""B3 — Are currently available metagraph signals enough?

For each miner at sampled snapshots, build block-level signals known at time t
(including axon / commitment updates, stake and shared coldkeys / IPs) and
predict what happens next:

  emission_change   change in log emission share over the next `pred_horizon_days`
  rank_change       change in percentile rank over the same horizon
  coordination      strongest co-movement with any other miner over the next
                    `pred_coord_days` (max correlation of log-share changes)
  sybil_onset       joins a B1 sybil-like community within the next `pred_coord_days`,
                    among miners not in one now (binary; R^2 is Efron's R^2 on
                    predicted probabilities; membership itself is too persistent to
                    be an informative target)
  emission_level    log share at t+h (reported, but persistence makes R^2 high
                    by construction, so it is not used for the verdict)

Improvements over a single R^2:
  * strictly out-of-sample: train on the first part of the period, test on the rest,
    with an embargo of one horizon so no training target overlaps the test period
  * a persistence baseline (each target's own history only), so incremental R^2
    shows what the metagraph adds beyond "what happened last"
  * linear (ridge) and non-linear (gradient boosting) models; the better one is the
    estimate of how much the signals can explain
  * drop-one-group ablation shows which signal families carry the information
  * results split by subnet and by mechanism type from B2

Low R^2 is evidence that the available signals are insufficient, but it can also
mean the outcome is mostly random (e.g. validator query sampling noise). The
persistence baseline and the per-group ablation help separate the two.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .subnet import SubnetData
from .support import validator_support

EPS = 1e-6

FEATURE_GROUPS: Dict[str, List[str]] = {
    "emission_history": ["log_share", "d_log_share_1", "d_log_share_mid", "d_log_share_h",
                         "vol_log_share", "pct_rank", "d_pct_rank_h"],
    "consensus_weights": ["log_cons_share", "d_log_cons_h", "emission_consensus_gap",
                          "log_median_support", "d_log_median_support_h", "support_breadth",
                          "d_support_breadth_h", "support_cv"],
    "bonds": ["log_bonds_share", "d_log_bonds_h"],
    "registration_activity": ["log_age_days", "is_new", "active", "sets_weights",
                              "axon_updated_h", "has_commitment", "commitment_updated_h"],
    "stake_identity": ["log_stake_share", "same_coldkey_peers", "same_ip24_peers"],
    "coordination": ["sync_now", "b1_member_now"],
    "subnet_context": ["ctx_gini_inc", "ctx_top10_share", "ctx_rank_corr", "ctx_validator_agreement",
                       "ctx_contested_share", "ctx_burn_share", "ctx_replacement_rate",
                       "ctx_n_validators", "ctx_copy_div_share", "ctx_log_n_miners"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]

TARGETS = {
    "emission_change": {"y": "y_emission_change", "baseline": ["d_log_share_h", "d_log_share_1"], "binary": False},
    "rank_change": {"y": "y_rank_change", "baseline": ["d_pct_rank_h"], "binary": False},
    "coordination": {"y": "y_coordination", "baseline": ["sync_now"], "binary": False},
    "sybil_onset": {"y": "y_sybil_onset", "baseline": ["sync_now"], "binary": True},
    "emission_level": {"y": "y_emission_level", "baseline": ["log_share"], "binary": False, "verdict": False},
}
CTX = {"gini_inc": "ctx_gini_inc", "top10_share": "ctx_top10_share", "rank_corr": "ctx_rank_corr",
       "validator_agreement": "ctx_validator_agreement", "contested_share": "ctx_contested_share",
       "burn_share": "ctx_burn_share", "replacement_rate": "ctx_replacement_rate",
       "n_validators": "ctx_n_validators", "copy_div_share": "ctx_copy_div_share"}


def _max_corr(D: np.ndarray) -> np.ndarray:
    """D: [w, n] changes. Max correlation of each column with any other column."""
    n = D.shape[1]
    if n < 2 or D.shape[0] < 4:
        return np.full(n, np.nan)
    Dc = D - D.mean(axis=0)
    sdv = np.linalg.norm(Dc, axis=0)
    ok = sdv > 0
    Dc = Dc / np.where(ok, sdv, 1)
    C = Dc.T @ Dc
    np.fill_diagonal(C, -np.inf)
    C[~ok, :] = -np.inf
    C[:, ~ok] = -np.inf
    out = C.max(axis=1)
    return np.where(np.isfinite(out), out, np.nan)


def build_rows(sd: SubnetData, panel: pd.DataFrame, cfg: Config, mech_type: str,
               b1_members: pd.DataFrame) -> pd.DataFrame:
    T, N = sd.T, sd.N
    h = sd.snaps(cfg.pred_horizon_days)
    hc = sd.snaps(cfg.pred_coord_days)
    Lh = sd.snaps(cfg.pred_history_days)
    mid = max(2, h // 2)
    step = max(1, h // 2)
    back = max(h, Lh)
    if T < back + max(h, hc) + 5:
        return pd.DataFrame()
    sup = validator_support(sd)
    miners = sd.is_miner
    n_m = np.maximum(miners.sum(axis=1), 1)
    s = sd.miner_emi_share
    ls = np.log(s + EPS)
    med = np.array([np.median(ls[t][miners[t]]) if miners[t].any() else 0.0 for t in range(T)])
    lsc = ls - med[:, None]                      # common-mode removed (for co-movement)
    dls = np.diff(lsc, axis=0, prepend=lsc[:1])
    pct = np.zeros((T, N))
    for t in range(T):
        m = miners[t]
        if m.sum() > 1:
            pct[t, m] = (pd.Series(s[t, m]).rank().to_numpy() - 1) / (m.sum() - 1)
    lcons = np.log(sup.cons_share + EPS)
    lmed = np.log(sup.median_share + EPS)
    lbond = np.log(sup.bonds_share + EPS)
    age_days = (sd.blocks[:, None] - sd.reg_block) * 12 / 86400 if sd.has_reg_block else np.full((T, N), np.nan)
    rep_cum = np.cumsum(sd.replaced, axis=0)
    lu = np.nan_to_num(sd.last_update)
    sets_w = np.zeros((T, N))
    sets_w[1:] = (lu[1:] > lu[:-1]) & ~sd.replaced[1:]

    ab, cb = np.nan_to_num(sd.axon_block), sd.commit_block
    ab_up = np.zeros((T, N))
    ab_up[1:] = ab[1:] > ab[:-1]
    cb_up = np.zeros((T, N))
    with np.errstate(invalid="ignore"):
        cb_up[1:] = np.nan_to_num(cb[1:]) > np.nan_to_num(cb[:-1])
    stake_sh = np.where(miners, np.nan_to_num(sd.stake), 0.0)
    stake_sh = np.log(stake_sh / np.maximum(stake_sh.sum(axis=1, keepdims=True), 1e-12) + EPS)
    has_stake = bool(np.isfinite(sd.stake).any())
    ck_arr, ip_arr = sd.coldkeys, sd.axon_ip
    has_ck, has_ip = bool((ck_arr != "").any()), bool((ip_arr != "").any())

    def peers(keys: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Number of other miners at t sharing the same non-empty key."""
        out = np.zeros(len(keys))
        k = pd.Series(keys[m])
        k = k[k != ""]
        counts = k.map(k.value_counts()).to_numpy() - 1
        out[np.where(m)[0][k.index.to_numpy()]] = counts
        return out

    syb = {}
    if len(b1_members) and "sybil_like_track" in b1_members:
        m = b1_members[b1_members["sybil_like_track"]]
        for b, g in m.groupby("t_end"):
            syb[int(b)] = set(g["occupant"])

    def in_sybil(occ: str, lo: int, hi: int) -> bool:
        return any(lo < b <= hi and occ in v for b, v in syb.items())

    p = panel.set_index("t")
    cut = int(T * cfg.pred_train_frac)
    rows = []
    for t in range(back, T - max(h, hc), step):
        valid = miners[t] & miners[t + h] & (rep_cum[t + max(h, hc)] == rep_cum[t - back])
        thr = cfg.min_miner_share_factor / n_m[t]
        valid &= np.maximum(s[t], s[t + h]) >= thr
        uids = np.where(valid)[0]
        if len(uids) < 3:
            continue
        sync_now = _max_corr(dls[t - Lh + 1:t + 1][:, uids])
        sync_fut = _max_corr(dls[t + 1:t + hc + 1][:, uids])
        vol = dls[t - Lh + 1:t + 1][:, uids].std(axis=0)
        ctx = p.loc[t]
        ck_peers = peers(ck_arr[t], miners[t]) if has_ck else None
        ip24 = np.array([".".join(x.split(".")[:3]) if x.count(".") == 3 else x for x in ip_arr[t]])
        ip_peers = peers(ip24, miners[t]) if has_ip else None
        split = "train" if t + max(h, hc) < cut else ("test" if t >= cut else "embargo")
        for j, u in enumerate(uids):
            occ = sd.occupant[t, u]
            r = {"netuid": sd.netuid, "t": t, "uid": int(u), "mechanism_type": mech_type, "split": split,
                 "log_share": ls[t, u], "d_log_share_1": ls[t, u] - ls[t - 1, u],
                 "d_log_share_mid": ls[t, u] - ls[t - mid, u], "d_log_share_h": ls[t, u] - ls[t - h, u],
                 "vol_log_share": vol[j], "pct_rank": pct[t, u], "d_pct_rank_h": pct[t, u] - pct[t - h, u],
                 "log_cons_share": lcons[t, u], "d_log_cons_h": lcons[t, u] - lcons[t - h, u],
                 "emission_consensus_gap": ls[t, u] - lcons[t, u],
                 "log_median_support": lmed[t, u], "d_log_median_support_h": lmed[t, u] - lmed[t - h, u],
                 "support_breadth": sup.breadth[t, u], "d_support_breadth_h": sup.breadth[t, u] - sup.breadth[t - h, u],
                 "support_cv": sup.cv[t, u],
                 "log_bonds_share": lbond[t, u], "d_log_bonds_h": lbond[t, u] - lbond[t - h, u],
                 "log_age_days": np.log1p(max(age_days[t, u], 0)) if np.isfinite(age_days[t, u]) else np.nan,
                 "is_new": float(age_days[t, u] < 1) if np.isfinite(age_days[t, u]) else np.nan,
                 "active": float(np.nan_to_num(sd.active[t, u])), "sets_weights": sets_w[t - h + 1:t + 1, u].max(),
                 "axon_updated_h": float(ab_up[t - h + 1:t + 1, u].max()),
                 "has_commitment": float(np.isfinite(cb[t, u])),
                 "commitment_updated_h": float(cb_up[t - h + 1:t + 1, u].max()),
                 "log_stake_share": stake_sh[t, u] if has_stake else np.nan,
                 "same_coldkey_peers": np.log1p(ck_peers[u]) if ck_peers is not None else np.nan,
                 "same_ip24_peers": np.log1p(ip_peers[u]) if ip_peers is not None else np.nan,
                 "sync_now": sync_now[j], "b1_member_now": float(in_sybil(occ, t - hc, t)),
                 "ctx_log_n_miners": np.log(n_m[t]),
                 "y_emission_change": ls[t + h, u] - ls[t, u], "y_emission_level": ls[t + h, u],
                 "y_rank_change": pct[t + h, u] - pct[t, u], "y_coordination": sync_fut[j],
                 "y_sybil_onset": (float(in_sybil(occ, t, t + hc)) if not in_sybil(occ, t - hc, t) else np.nan)}
            for k, c in CTX.items():
                r[c] = float(ctx[k]) if k in ctx else np.nan
            rows.append(r)
    df = pd.DataFrame(rows)
    df = df[df["split"] != "embargo"]
    if len(df) > cfg.pred_max_rows_per_subnet:
        df = df.sample(cfg.pred_max_rows_per_subnet, random_state=cfg.seed)
    return df


def _fit_predict(kind: str, binary: bool, Xtr, ytr, Xte, seed: int):
    if kind == "ridge":
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        Xte = np.where(np.isfinite(Xte), Xte, med)
        if binary:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
            m.fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1]
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    if binary:
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def _score(y, pred, binary: bool) -> Dict[str, float]:
    out = {"r2": float(r2_score(y, np.clip(pred, 0, 1) if binary else pred)) if len(y) > 1 and np.var(y) > 0 else np.nan}
    if binary:
        out["auc"] = float(roc_auc_score(y, pred)) if 0 < y.sum() < len(y) else np.nan
    return out


def run(rows: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    if rows is None or not len(rows):
        return {}
    feats = [f for f in ALL_FEATURES if f in rows and rows[f].notna().any()]
    groups = {g: [f for f in fs if f in feats] for g, fs in FEATURE_GROUPS.items()}
    res, by_sub, by_mech, abl = [], [], [], []
    for name, spec in TARGETS.items():
        d = rows[np.isfinite(rows[spec["y"]].to_numpy(float))]
        tr, te = d[d["split"] == "train"], d[d["split"] == "test"]
        if len(tr) > cfg.pred_max_train_rows:
            tr = tr.sample(cfg.pred_max_train_rows, random_state=cfg.seed)
        binary = spec["binary"]
        too_few = len(tr) < 50 or len(te) < 50
        if binary and not too_few:
            too_few = min(tr[spec["y"]].sum(), te[spec["y"]].sum()) < 5
        if too_few:
            res.append({"target": name, "note": "not enough rows (or < 5 positive cases in train or test)",
                        "n_train": len(tr), "n_test": len(te)})
            continue
        ytr, yte = tr[spec["y"]].to_numpy(float), te[spec["y"]].to_numpy(float)
        preds = {}
        for fs_name, fs in (("persistence_baseline", [f for f in spec["baseline"] if f in feats]), ("all_signals", feats)):
            if not fs:
                continue
            for kind in ("ridge", "boosting"):
                pr = _fit_predict(kind, binary, tr[fs].to_numpy(float), ytr, te[fs].to_numpy(float), cfg.seed)
                preds[(fs_name, kind)] = pr
                res.append({"target": name, "feature_set": fs_name, "model": kind,
                            "n_train": len(tr), "n_test": len(te), **_score(yte, pr, binary)})
        best = max((k for k in preds if k[0] == "all_signals"),
                   key=lambda k: _score(yte, preds[k], binary)["r2"])
        pr = preds[best]
        for (net, g) in te.assign(_p=pr).groupby("netuid"):
            if len(g) >= 30:
                by_sub.append({"target": name, "netuid": net, "mechanism_type": g["mechanism_type"].iloc[0],
                               "n_test": len(g), **_score(g[spec["y"]].to_numpy(float), g["_p"].to_numpy(), binary)})
        for (mt, g) in te.assign(_p=pr).groupby("mechanism_type"):
            by_mech.append({"target": name, "mechanism_type": mt, "n_test": len(g), "subnets": g["netuid"].nunique(),
                            **_score(g[spec["y"]].to_numpy(float), g["_p"].to_numpy(), binary)})
        full = _score(yte, pr, binary)["r2"]
        for gname, gf in groups.items():
            keep = [f for f in feats if f not in gf]
            if not gf or not keep:
                continue
            p2 = _fit_predict(best[1], binary, tr[keep].to_numpy(float), ytr, te[keep].to_numpy(float), cfg.seed)
            abl.append({"target": name, "dropped_group": gname, "features_dropped": len(gf),
                        "r2_without": _score(yte, p2, binary)["r2"], "r2_full": full,
                        "r2_loss": full - _score(yte, p2, binary)["r2"]})
    rdf = pd.DataFrame(res)

    verdict = []
    for name, spec in TARGETS.items():
        g = rdf[(rdf.get("target") == name)] if len(rdf) else rdf
        if not len(g) or "r2" not in g or g["r2"].isna().all():
            continue
        full = g.loc[g["feature_set"] == "all_signals", "r2"].max()
        base = g.loc[g["feature_set"] == "persistence_baseline", "r2"].max()
        v = ("insufficient" if full < cfg.r2_insufficient else
             "weak" if full < cfg.r2_adequate else "adequate")
        verdict.append({"target": name, "best_r2_all_signals": full, "best_r2_persistence": base,
                        "incremental_r2": full - base if np.isfinite(base) else np.nan,
                        "verdict": v if spec.get("verdict", True) else "not used (persistence-dominated)"})
    vdf = pd.DataFrame(verdict)
    used = vdf[~vdf["verdict"].str.startswith("not used")] if len(vdf) else vdf
    overall = pd.DataFrame([{
        "targets_evaluated": len(used),
        "targets_insufficient": int((used["verdict"] == "insufficient").sum()) if len(used) else 0,
        "targets_weak": int((used["verdict"] == "weak").sum()) if len(used) else 0,
        "median_r2_all_signals": float(used["best_r2_all_signals"].median()) if len(used) else np.nan,
        "median_incremental_r2": float(used["incremental_r2"].median()) if len(used) else np.nan,
        "rows": len(rows), "features": len(feats),
    }])
    return {"b3_results": rdf, "b3_verdict": vdf, "b3_summary": overall, "b3_by_subnet": pd.DataFrame(by_sub),
            "b3_by_mechanism": pd.DataFrame(by_mech), "b3_ablation": pd.DataFrame(abl)}
