"""Tolerant parsers for collector values (numbers, lists, sparse weight rows, hotkeys)."""
from typing import List, Optional, Tuple

import numpy as np


def _is_num(x) -> bool:
    return isinstance(x, (int, float, bool, np.integer, np.floating)) and not isinstance(x, str)


def unwrap(x):
    """Strip single-element list wrappers: [[1,2,3]] -> [1,2,3], [5] -> 5."""
    while isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        x = x[0]
    return x


def hk_to_str(x) -> Optional[str]:
    """Hotkeys/coldkeys may decode as ss58 strings, hex strings, or byte lists."""
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("ss58", "address", "hotkey", "value"):
            if k in x:
                return hk_to_str(x[k])
        return None
    if isinstance(x, list):
        x = unwrap(x)
        if len(x) > 0 and all(_is_num(v) for v in x):
            return "0x" + bytes(int(v) & 0xFF for v in x).hex()
        if len(x) == 1:
            return hk_to_str(x[0])
    return str(x)


def parse_scalar(x) -> float:
    """Subnet-level value -> float (NaN if missing)."""
    if x is None:
        return np.nan
    if isinstance(x, list):
        x = unwrap(x)
        if isinstance(x, list):
            if len(x) == 0:
                return np.nan
            x = x[0]
    if isinstance(x, dict):
        for k in ("value", "bits"):
            if k in x:
                return parse_scalar(x[k])
        return np.nan
    if isinstance(x, bool):
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def parse_vector(x) -> Tuple[np.ndarray, str]:
    """Per-UID vector. Accepts [v0, v1, ...] or [[uid, v], ...]."""
    if x is None:
        return np.array([], dtype=float), "missing"
    x = unwrap(x)
    if not isinstance(x, list):
        v = parse_scalar(x)
        return np.array([v]), "scalar"
    if len(x) == 0:
        return np.array([], dtype=float), "empty"
    a = _try_array(x)
    if a is not None and a.ndim == 1:
        return a, "dense"
    if all(_is_num(v) or v is None for v in x):
        return np.array([np.nan if v is None else float(v) for v in x]), "dense"
    if all(isinstance(v, list) and len(v) == 2 and _is_num(v[0]) for v in x):
        n = int(max(v[0] for v in x)) + 1
        out = np.full(n, np.nan)
        for uid, val in x:
            out[int(uid)] = parse_scalar(val)
        return out, "pairs"
    return np.array([parse_scalar(v) for v in x]), "mixed"


def parse_string_vector(x) -> Tuple[List[Optional[str]], str]:
    """Per-UID strings (Keys -> hotkeys). Accepts [hk0, ...] or [[uid, hk], ...]."""
    if x is None:
        return [], "missing"
    if isinstance(x, dict):
        n = max(int(k) for k in x) + 1 if x else 0
        out = [None] * n
        for k, v in x.items():
            out[int(k)] = hk_to_str(v)
        return out, "dict"
    if not isinstance(x, list) or len(x) == 0:
        return [], "empty"
    first = x[0]
    if isinstance(first, list) and len(first) == 2 and _is_num(first[0]) and not all(_is_num(v) for v in first):
        n = int(max(v[0] for v in x)) + 1
        out = [None] * n
        for uid, hk in x:
            out[int(uid)] = hk_to_str(hk)
        return out, "pairs"
    return [hk_to_str(v) for v in x], "dense"


def _try_array(x, dtype=np.float64):
    try:
        return np.asarray(x, dtype=dtype)
    except (ValueError, TypeError):
        return None


def parse_matrix(x, n: int) -> Tuple[np.ndarray, str]:
    """Weights/Bonds -> dense [n, n] float32 (rows = source UID).

    Accepted layouts:
      dense      : [[w00, w01, ...], ...]
      sparse_rows: [[[dest, w], ...], ...]          (row index = source uid)
      keyed_rows : [[src, [[dest, w], ...]], ...]
    """
    m = np.zeros((n, n), dtype=np.float32)
    if x is None:
        return m, "missing"
    if not isinstance(x, list) or len(x) == 0:
        return m, "empty"

    def fill_row(i, row):
        if i >= n or not isinstance(row, list) or len(row) == 0:
            return
        r = _try_array(row)
        if r is not None and r.ndim == 1:                 # dense row
            k = min(len(r), n)
            m[i, :k] = r[:k]
        elif r is not None and r.ndim == 2 and r.shape[1] == 2:  # [[dest, w], ...]
            j = r[:, 0].astype(int)
            ok = (j >= 0) & (j < n)
            m[i, j[ok]] = r[ok, 1]
        else:                                             # mixed / decoded objects
            for pair in row:
                if isinstance(pair, list) and len(pair) == 2 and _is_num(pair[0]):
                    j = int(pair[0])
                    if 0 <= j < n:
                        m[i, j] = float(parse_scalar(pair[1]))

    first = next((r for r in x if isinstance(r, list) and len(r) > 0), None)
    if first is None:
        return m, "empty"
    if len(first) == 2 and _is_num(first[0]) and isinstance(first[1], list):
        for src, row in x:
            fill_row(int(src), row)
        return m, "keyed_rows"
    a = _try_array(x, np.float32)
    if a is not None and a.ndim == 2:
        k0, k1 = min(a.shape[0], n), min(a.shape[1], n)
        m[:k0, :k1] = a[:k0, :k1]
        return m, "dense"
    layout = "dense" if all(_is_num(v) for v in first) else "sparse_rows"
    for i, row in enumerate(x):
        fill_row(i, row)
    return m, layout
