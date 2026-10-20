"""Statistical helpers: concentration, rank correlation, change points, AUC, graph components."""
from typing import List, Tuple

import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def gini(x: np.ndarray) -> float:
    x = np.sort(np.nan_to_num(np.asarray(x, dtype=float)))
    if x.sum() <= 0 or len(x) == 0:
        return np.nan
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def top_k_share(x: np.ndarray, k: int = 10) -> float:
    x = np.nan_to_num(np.asarray(x, dtype=float))
    s = x.sum()
    return float(np.sort(x)[-k:].sum() / s) if s > 0 else np.nan


def hhi(x: np.ndarray) -> float:
    x = np.nan_to_num(np.asarray(x, dtype=float))
    s = x.sum()
    return float(((x / s) ** 2).sum()) if s > 0 else np.nan


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.ptp(a[m]) == 0 or np.ptp(b[m]) == 0:
        return np.nan
    return float(stats.spearmanr(a[m], b[m]).statistic)


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Xn = X / n
    return Xn @ Xn.T


def corr_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation between rows; constant rows get 0 correlation."""
    Xc = X - X.mean(axis=1, keepdims=True)
    sd = np.linalg.norm(Xc, axis=1, keepdims=True)
    const = (sd[:, 0] == 0)
    sd[sd == 0] = 1.0
    C = (Xc / sd) @ (Xc / sd).T
    C[const, :] = 0
    C[:, const] = 0
    return C


def components(adj: np.ndarray, min_size: int) -> List[np.ndarray]:
    adj = adj.copy()
    np.fill_diagonal(adj, False)
    _, labels = connected_components(csr_matrix(adj), directed=False)
    out = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) >= min_size:
            out.append(idx)
    return out


def _seg_cost(cs: np.ndarray, cs2: np.ndarray, i: int, j: int) -> float:
    n = j - i
    s = cs[j] - cs[i]
    s2 = cs2[j] - cs2[i]
    return float((s2 - (s * s) / n).sum())


def change_points(X: np.ndarray, min_size: int, penalty: float,
                  min_effect: float) -> List[int]:
    """Binary segmentation for mean shifts in a [T, d] series.

    Columns are z-scored first. A split is accepted when it reduces the L2 cost by
    more than `penalty` and the adjacent segment means differ by at least
    `min_effect` (standardized units, max over dimensions).
    """
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(X).all(axis=1)
    idx_map = np.where(ok)[0]
    X = X[ok]
    T = len(X)
    if T < 2 * min_size:
        return []
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - X.mean(axis=0)) / sd
    cs = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)])
    cs2 = np.vstack([np.zeros(X.shape[1]), np.cumsum(X * X, axis=0)])

    cps: List[int] = []
    stack = [(0, T)]
    while stack:
        i, j = stack.pop()
        if j - i < 2 * min_size:
            continue
        base = _seg_cost(cs, cs2, i, j)
        best, best_k = 0.0, None
        for k in range(i + min_size, j - min_size + 1):
            gain = base - _seg_cost(cs, cs2, i, k) - _seg_cost(cs, cs2, k, j)
            if gain > best:
                best, best_k = gain, k
        if best_k is None or best <= penalty:
            continue
        left = (cs[best_k] - cs[i]) / (best_k - i)
        right = (cs[j] - cs[best_k]) / (j - best_k)
        if np.max(np.abs(right - left)) < min_effect:
            continue
        cps.append(best_k)
        stack += [(i, best_k), (best_k, j)]
    return sorted(int(idx_map[k]) for k in cps)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    m = np.isfinite(scores)
    s, y = scores[m], labels[m].astype(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    r = stats.rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def trailing_z(x: np.ndarray, window: int) -> np.ndarray:
    """Robust z-score of x[t] against the trailing window x[t-window:t]."""
    x = np.asarray(x, dtype=float)
    z = np.full(len(x), np.nan)
    for t in range(window, len(x)):
        base = x[t - window:t]
        base = base[np.isfinite(base)]
        if len(base) < max(3, window // 2) or not np.isfinite(x[t]):
            continue
        med = np.median(base)
        mad = np.median(np.abs(base - med)) * 1.4826
        if mad == 0:
            mad = base.std() or 1e-9
        z[t] = (x[t] - med) / mad
    return z


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> Tuple[float, float]:
    """Spearman correlation of x and y after removing the rank-linear effect of control."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    if m.sum() < 6:
        return np.nan, np.nan
    rx, ry, rc = (stats.rankdata(v[m]) for v in (x, y, control))
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return np.nan, np.nan
    if np.ptp(rc) == 0:  # nothing to control for
        r = stats.spearmanr(rx, ry)
        return float(r.statistic), float(r.pvalue)
    res = []
    for r in (rx, ry):
        beta = np.polyfit(rc, r, 1)
        res.append(r - np.polyval(beta, rc))
    if np.ptp(res[0]) == 0 or np.ptp(res[1]) == 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(res[0], res[1])
    return float(r), float(p)
