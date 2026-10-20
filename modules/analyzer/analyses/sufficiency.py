"""Are the currently available metagraph signals enough?

Builds block-level signals per miner at sampled snapshots and predicts, out of sample:
  emission_change   change in log emission share over the next horizon
  rank_change       change in percentile rank
  coordination      strongest co-movement with any other miner over the next window
  sybil_onset       joining a sybil-like community soon (binary; Efron's R^2 + AUC)
  emission_level    log share at t+h (persistence-dominated; reported, not used for the verdict)
Each target is compared with a persistence baseline (its own recent history), fit with
ridge and gradient boosting, and dropped one signal group at a time (ablation).
Low R^2 beyond persistence = the signals don't explain what happens next.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import Analysis

EPS = 1e-6

FEATURE_GROUPS: Dict[str, List[str]] = {
    "emission_history": ["log_share", "d_log_share_1", "d_log_share_mid", "d_log_share_h", "vol_log_share",
                         "pct_rank", "d_pct_rank_h"],
    "consensus_weights": ["log_cons_share", "d_log_cons_h", "emission_consensus_gap", "log_median_support",
                          "d_log_median_support_h", "support_breadth", "d_support_breadth_h", "support_cv"],
    "validator_stake": ["log_stake_weighted_support", "top_validator_support_share"],
    "registration_activity": ["log_age_days", "is_new", "active", "sets_weights"],
    "coordination": ["sync_now", "sybil_member_now"],
    "subnet_context": ["ctx_gini_inc", "ctx_top10_share", "ctx_rank_corr", "ctx_validator_agreement",
                       "ctx_contested_share", "ctx_burn_share", "ctx_replacement_rate", "ctx_n_validators",
                       "ctx_copy_div_share", "ctx_stake_concentration", "ctx_log_n_miners"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
TARGETS = {
    "emission_change": {"y": "y_emission_change", "baseline": ["d_log_share_h", "d_log_share_1"], "binary": False},
    "rank_change": {"y": "y_rank_change", "baseline": ["d_pct_rank_h"], "binary": False},
    "coordination": {"y": "y_coordination", "baseline": ["sync_now"], "binary": False},
    "sybil_onset": {"y": "y_sybil_onset", "baseline": ["sync_now"], "binary": True},
    "emission_level": {"y": "y_emission_level", "baseline": ["log_share"], "binary": False, "verdict": False},
}
CTX = ["gini_inc", "top10_share", "rank_corr", "validator_agreement", "contested_share", "burn_share",
       "replacement_rate", "n_validators", "copy_div_share", "stake_concentration"]


def _max_corr(D: np.ndarray) -> np.ndarray:
    """D: [w, n] changes -> each column's max correlation with any other column."""
    n = D.shape[1]
    if n < 2 or D.shape[0] < 4:
        return np.full(n, np.nan)
    Dc = D - D.mean(axis=0)
    sd = np.linalg.norm(Dc, axis=0)
    ok = sd > 0
    Dc = Dc / np.where(ok, sd, 1)
    C = Dc.T @ Dc
    np.fill_diagonal(C, -np.inf)
    C[~ok, :] = -np.inf
    C[:, ~ok] = -np.inf
    out = C.max(axis=1)
    return np.where(np.isfinite(out), out, np.nan)


def _fit_predict(kind: str, binary: bool, Xtr, ytr, Xte, seed: int):
    if kind == "ridge":
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr, Xte = np.where(np.isfinite(Xtr), Xtr, med), np.where(np.isfinite(Xte), Xte, med)
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500) if binary else Ridge(alpha=1.0))
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1] if binary else m.predict(Xte)
    m = (HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor)(
        max_iter=200, learning_rate=0.1, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1] if binary else m.predict(Xte)


def _score(y, pred, binary: bool) -> Dict[str, float]:
    out = {"r2": float(r2_score(y, np.clip(pred, 0, 1) if binary else pred)) if len(y) > 1 and np.var(y) > 0 else np.nan}
    if binary:
        out["auc"] = float(roc_auc_score(y, pred)) if 0 < y.sum() < len(y) else np.nan
    return out


class SufficiencyAnalysis(Analysis):
    key = "sufficiency"
    title = "Are the available metagraph signals enough?"
    problem = ""
    questions = {
        "Q1": "How much of future emission change, rank change and coordination do the signals explain (out-of-sample R^2)?",
        "Q2": "How much do they add beyond each target's own history (incremental R^2)?",
        "Q3": "Which signal groups carry the information, and does it differ by mechanism type?",
    }
    tables = {
        "results": ("Q1-Q2", "per target x feature set x model: out-of-sample R^2 (and AUC for binary)"),
        "verdict": ("Q1-Q2", "per target: best R^2, persistence R^2, incremental R^2, verdict"),
        "summary": ("Q1", "targets judged insufficient / weak / adequate"),
        "by_subnet": ("Q3", "per target and subnet R^2 of the best model"),
        "by_mechanism": ("Q3", "per target and mechanism type R^2"),
        "ablation": ("Q3", "R^2 lost when each signal group is removed"),
    }

    def per_subnet(self, sn, panel, ctx):
        c, g = self.cfg.sufficiency, self.cfg.general
        T, N = sn.T, sn.N
        h, hc, Lh = sn.snaps(c.horizon_days), sn.snaps(c.coord_days), sn.snaps(c.history_days)
        mid, step, back = max(2, h // 2), max(1, h // 2), max(h, Lh)
        if T < back + max(h, hc) + 5:
            return {}
        mech = ctx.get("mechanism_type", "unknown")
        sup = sn.support
        miners = sn.is_miner
        n_m = np.maximum(miners.sum(axis=1), 1)
        s = sn.miner_emi_share
        ls = np.log(s + EPS)
        med = np.array([np.median(ls[t][miners[t]]) if miners[t].any() else 0.0 for t in range(T)])
        dls = np.diff(ls - med[:, None], axis=0, prepend=(ls - med[:, None])[:1])
        pct = np.zeros((T, N))
        for t in range(T):
            m = miners[t]
            if m.sum() > 1:
                pct[t, m] = (pd.Series(s[t, m]).rank().to_numpy() - 1) / (m.sum() - 1)
        lcons, lmed = np.log(sup.cons_share + EPS), np.log(sup.median_share + EPS)
        age = (sn.blocks[:, None] - sn.reg_block) * 12 / 86400 if sn.has_reg_block else np.full((T, N), np.nan)
        rep_cum = np.cumsum(sn.replaced, axis=0)
        lu = np.nan_to_num(sn.last_update)
        sets_w = np.zeros((T, N))
        sets_w[1:] = (lu[1:] > lu[:-1]) & ~sn.replaced[1:]
        members = ctx.get("sybil_members", pd.DataFrame())
        syb = {}
        if len(members) and "sybil_like_track" in members:
            for b, gr in members[members["sybil_like_track"]].groupby("t_end"):
                syb[int(b)] = set(gr["occupant"])

        def in_sybil(occ, lo, hi):
            return any(lo < b <= hi and occ in v for b, v in syb.items())

        p = panel.set_index("t")
        cut = int(T * c.train_frac)
        rows = []
        for t in range(back, T - max(h, hc), step):
            valid = miners[t] & miners[t + h] & (rep_cum[t + max(h, hc)] == rep_cum[t - back])
            valid &= np.maximum(s[t], s[t + h]) >= g.min_miner_share_factor / n_m[t]
            uids = np.where(valid)[0]
            if len(uids) < 3:
                continue
            sync_now = _max_corr(dls[t - Lh + 1:t + 1][:, uids])
            sync_fut = _max_corr(dls[t + 1:t + hc + 1][:, uids])
            vol = dls[t - Lh + 1:t + 1][:, uids].std(axis=0)
            # stake-weighted support and dependence on the largest validator
            vr = sn.active_validator_rows(t)
            sws, topshare = np.full(N, np.nan), np.full(N, np.nan)
            if len(vr):
                Wm = sn.W[t][vr]
                w = sn.validator_weight(t, sn.weights_rows[vr])
                w = w / (w.sum() or 1)
                contrib = Wm * w[:, None]
                tot = contrib.sum(axis=0)
                sws = np.log(tot / (tot[miners[t]].sum() or 1) + EPS)
                with np.errstate(invalid="ignore", divide="ignore"):
                    topshare = np.where(tot > 0, contrib.max(axis=0) / tot, np.nan)
            ctx_row = p.loc[t]
            split = "train" if t + max(h, hc) < cut else ("test" if t >= cut else "embargo")
            for j, u in enumerate(uids):
                occ = sn.occupant[t, u]
                now = in_sybil(occ, t - hc, t)
                r = {"netuid": sn.netuid, "t": t, "uid": int(u), "mechanism_type": mech, "split": split,
                     "log_share": ls[t, u], "d_log_share_1": ls[t, u] - ls[t - 1, u],
                     "d_log_share_mid": ls[t, u] - ls[t - mid, u], "d_log_share_h": ls[t, u] - ls[t - h, u],
                     "vol_log_share": vol[j], "pct_rank": pct[t, u], "d_pct_rank_h": pct[t, u] - pct[t - h, u],
                     "log_cons_share": lcons[t, u], "d_log_cons_h": lcons[t, u] - lcons[t - h, u],
                     "emission_consensus_gap": ls[t, u] - lcons[t, u],
                     "log_median_support": lmed[t, u], "d_log_median_support_h": lmed[t, u] - lmed[t - h, u],
                     "support_breadth": sup.breadth[t, u], "d_support_breadth_h": sup.breadth[t, u] - sup.breadth[t - h, u],
                     "support_cv": sup.cv[t, u],
                     "log_stake_weighted_support": sws[u], "top_validator_support_share": topshare[u],
                     "log_age_days": np.log1p(max(age[t, u], 0)) if np.isfinite(age[t, u]) else np.nan,
                     "is_new": float(age[t, u] < 1) if np.isfinite(age[t, u]) else np.nan,
                     "active": float(np.nan_to_num(sn.active[t, u])), "sets_weights": sets_w[t - h + 1:t + 1, u].max(),
                     "sync_now": sync_now[j], "sybil_member_now": float(now),
                     "ctx_log_n_miners": np.log(n_m[t]),
                     "y_emission_change": ls[t + h, u] - ls[t, u], "y_emission_level": ls[t + h, u],
                     "y_rank_change": pct[t + h, u] - pct[t, u], "y_coordination": sync_fut[j],
                     "y_sybil_onset": np.nan if now else float(in_sybil(occ, t, t + hc))}
                for k in CTX:
                    r[f"ctx_{k}"] = float(ctx_row[k]) if k in ctx_row else np.nan
                rows.append(r)
        df = pd.DataFrame(rows)
        if len(df):
            df = df[df["split"] != "embargo"]
            if len(df) > c.max_rows_per_subnet:
                df = df.sample(c.max_rows_per_subnet, random_state=self.cfg.seed)
        return {"_rows": df}

    def aggregate(self, results, panel):
        c = self.cfg.sufficiency
        rows = self.combined("_rows")
        if not len(rows):
            return {}
        feats = [f for f in ALL_FEATURES if f in rows and rows[f].notna().any()]
        groups = {gname: [f for f in fs if f in feats] for gname, fs in FEATURE_GROUPS.items()}
        res, by_sub, by_mech, abl = [], [], [], []
        for name, spec in TARGETS.items():
            d = rows[np.isfinite(rows[spec["y"]].to_numpy(float))]
            tr, te = d[d["split"] == "train"], d[d["split"] == "test"]
            if len(tr) > c.max_train_rows:
                tr = tr.sample(c.max_train_rows, random_state=self.cfg.seed)
            binary = spec["binary"]
            too_few = len(tr) < 50 or len(te) < 50 or (binary and min(tr[spec["y"]].sum(), te[spec["y"]].sum()) < 5)
            if too_few:
                res.append({"target": name, "note": "not enough rows (or < 5 positive cases in train or test)",
                            "n_train": len(tr), "n_test": len(te)})
                continue
            ytr, yte = tr[spec["y"]].to_numpy(float), te[spec["y"]].to_numpy(float)
            preds = {}
            for fs_name, fs in (("persistence_baseline", [f for f in spec["baseline"] if f in feats]), ("all_signals", feats)):
                for kind in (("ridge", "boosting") if fs else ()):
                    pr = _fit_predict(kind, binary, tr[fs].to_numpy(float), ytr, te[fs].to_numpy(float), self.cfg.seed)
                    preds[(fs_name, kind)] = pr
                    res.append({"target": name, "feature_set": fs_name, "model": kind, "n_train": len(tr),
                                "n_test": len(te), **_score(yte, pr, binary)})
            best = max((k for k in preds if k[0] == "all_signals"), key=lambda k: _score(yte, preds[k], binary)["r2"])
            pr = preds[best]
            te2 = te.assign(_p=pr)
            for net, gr in te2.groupby("netuid"):
                if len(gr) >= 30:
                    by_sub.append({"target": name, "netuid": net, "mechanism_type": gr["mechanism_type"].iloc[0],
                                   "n_test": len(gr), **_score(gr[spec["y"]].to_numpy(float), gr["_p"].to_numpy(), binary)})
            for mt, gr in te2.groupby("mechanism_type"):
                by_mech.append({"target": name, "mechanism_type": mt, "n_test": len(gr), "subnets": gr["netuid"].nunique(),
                                **_score(gr[spec["y"]].to_numpy(float), gr["_p"].to_numpy(), binary)})
            full = _score(yte, pr, binary)["r2"]
            for gname, gf in groups.items():
                keep = [f for f in feats if f not in gf]
                if gf and keep:
                    r2 = _score(yte, _fit_predict(best[1], binary, tr[keep].to_numpy(float), ytr,
                                                  te[keep].to_numpy(float), self.cfg.seed), binary)["r2"]
                    abl.append({"target": name, "dropped_group": gname, "features_dropped": len(gf),
                                "r2_without": r2, "r2_full": full, "r2_loss": full - r2})
        rdf = pd.DataFrame(res)
        verdict = []
        for name, spec in TARGETS.items():
            gr = rdf[rdf["target"] == name] if len(rdf) else rdf
            if not len(gr) or "r2" not in gr or gr["r2"].isna().all():
                continue
            full = gr.loc[gr["feature_set"] == "all_signals", "r2"].max()
            base = gr.loc[gr["feature_set"] == "persistence_baseline", "r2"].max()
            v = "insufficient" if full < c.r2_insufficient else "weak" if full < c.r2_adequate else "adequate"
            verdict.append({"target": name, "best_r2_all_signals": full, "best_r2_persistence": base,
                            "incremental_r2": full - base if np.isfinite(base) else np.nan,
                            "verdict": v if spec.get("verdict", True) else "not used (persistence-dominated)"})
        vdf = pd.DataFrame(verdict)
        used = vdf[~vdf["verdict"].str.startswith("not used")] if len(vdf) else vdf
        return {"results": rdf, "verdict": vdf, "by_subnet": pd.DataFrame(by_sub),
                "by_mechanism": pd.DataFrame(by_mech), "ablation": pd.DataFrame(abl),
                "summary": pd.DataFrame([{
                    "targets_evaluated": len(used),
                    "targets_insufficient": int((used["verdict"] == "insufficient").sum()) if len(used) else 0,
                    "targets_weak": int((used["verdict"] == "weak").sum()) if len(used) else 0,
                    "median_r2_all_signals": float(used["best_r2_all_signals"].median()) if len(used) else np.nan,
                    "median_incremental_r2": float(used["incremental_r2"].median()) if len(used) else np.nan,
                    "rows": len(rows), "features": len(feats)}])}
