"""Step 1: raw snapshot JSON -> one compressed .npz per subnet.

Reads the file written by `MetagraphManager.cache_blocksnapshots` directly (not via
`load_blocksnapshots`, whose fixed constructor would drop the new fields). Each
raw entry holds one block and, for every field, a list aligned with `netuid`.
"""
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from .adapters import (hk_to_str, parse_axons, parse_certificates, parse_matrix,
                       parse_scalar, parse_string_vector, parse_vector)
from .config import HYPERPARAMS, RAW_FIELDS, Config

VECTOR_FIELDS = ["incentive", "emission", "dividends", "consensus", "vtrust",
                 "active", "vpermit", "last_update", "reg_block",
                 "axon_block", "prometheus_block", "commit_block", "stake", "stake_weight"]
SCALAR_FIELDS = HYPERPARAMS + ["subnet_tao", "alpha_in", "yc3", "commit_reveal", "moving_price"]
STRING_VECTOR_FIELDS = ["coldkeys", "axon_ip"]
MATRIX_FIELDS = ["weights", "bonds"]


def _load_entries(path_or_glob: str) -> List[dict]:
    paths = sorted(glob.glob(path_or_glob)) if any(c in path_or_glob for c in "*?[") else [path_or_glob]
    entries = []
    for p in paths:
        with open(p) as f:
            entries.extend(json.load(f))
    # de-duplicate by block number, keep the last occurrence
    by_block = {}
    for e in entries:
        by_block[int(e["block_info"]["number"])] = e
    return [by_block[b] for b in sorted(by_block)]


def _field_for(entry: dict, raw_key: str, idx: int, single: bool):
    if raw_key not in entry:
        return None
    v = entry[raw_key]
    if single:
        return v
    if isinstance(v, list) and idx < len(v):
        return v[idx]
    return None


def prepare(cfg: Config, verbose: bool = True) -> Dict:
    entries = _load_entries(cfg.raw_snapshots)
    if not entries:
        raise ValueError(f"No snapshots found at {cfg.raw_snapshots}")
    os.makedirs(cfg.processed_dir, exist_ok=True)

    # entries carry either a list of netuids (one entry per block) or a single netuid
    per_netuid: Dict[int, List[tuple]] = defaultdict(list)
    for e in entries:
        nets = e["netuid"]
        if isinstance(nets, list):
            for i, n in enumerate(nets):
                per_netuid[int(n)].append((e, i, False))
        else:
            per_netuid[int(nets)].append((e, 0, True))

    identity = _load_identity(cfg.identity_file) if cfg.identity_file else {}
    layouts: Dict[str, Counter] = defaultdict(Counter)
    meta = {"netuids": [], "netuid_order": [], "n_blocks": len(entries),
            "blocks": [int(e["block_info"]["number"]) for e in entries]}
    first_nets = entries[0]["netuid"]
    meta["netuid_order"] = [int(n) for n in first_nets] if isinstance(first_nets, list) else []

    for netuid, rows in sorted(per_netuid.items()):
        T = len(rows)
        blocks = np.array([int(e["block_info"]["number"]) for e, _, _ in rows])
        stamps = np.array([str(e["block_info"].get("timestamp", "")) for e, _, _ in rows])

        vecs = {k: [] for k in VECTOR_FIELDS}
        for e, i, single in rows:
            for k in VECTOR_FIELDS:
                v, lay = parse_vector(_field_for(e, RAW_FIELDS[k], i, single))
                vecs[k].append(v)
                layouts[k][lay] += 1
        N = max(len(v) for v in vecs["incentive"]) if vecs["incentive"] else 0
        if N == 0:
            continue

        arrays = {}
        for k in VECTOR_FIELDS:
            a = np.full((T, N), np.nan)
            for t, v in enumerate(vecs[k]):
                a[t, :min(len(v), N)] = v[:N]
            arrays[k] = a
        n_uids = np.array([len(v) for v in vecs["incentive"]])

        for k in SCALAR_FIELDS:
            arrays[k] = np.array([parse_scalar(_field_for(e, RAW_FIELDS[k], i, s)) for e, i, s in rows])

        hot = np.full((T, N), "", dtype=object)
        owner = np.full(T, "", dtype=object)
        cert_count = np.zeros(T)
        cert_flag = np.zeros((T, N), dtype=bool)
        strings = {k: np.full((T, N), "", dtype=object) for k in STRING_VECTOR_FIELDS}
        for t, (e, i, s) in enumerate(rows):
            for k in STRING_VECTOR_FIELDS:
                vals, _ = parse_string_vector(_field_for(e, RAW_FIELDS[k], i, s))
                for u, v in enumerate(vals[:N]):
                    strings[k][t, u] = v or ""
            hks, lay = parse_string_vector(_field_for(e, RAW_FIELDS["hotkeys"], i, s))
            layouts["hotkeys"][lay] += 1
            for u, hk in enumerate(hks[:N]):
                hot[t, u] = hk or ""
            owner[t] = hk_to_str(_field_for(e, RAW_FIELDS["owner_hotkey"], i, s)) or ""
            cnt, cert_hks, lay = parse_certificates(_field_for(e, RAW_FIELDS["certificates"], i, s))
            layouts["certificates"][lay] += 1
            cert_count[t] = cnt
            if cert_hks:
                hk_set = set(cert_hks)
                cert_flag[t] = [h in hk_set for h in hot[t]]

        # matrices: keep only rows that are ever non-zero (validator rows)
        for k in MATRIX_FIELDS:
            mats = []
            row_union = set()
            for t, (e, i, s) in enumerate(rows):
                m, lay = parse_matrix(_field_for(e, RAW_FIELDS[k], i, s), N)
                layouts[k][lay] += 1
                nz = np.where(m.sum(axis=1) > 0)[0]
                row_union.update(nz.tolist())
                mats.append({int(r): m[r].copy() for r in nz})
            rows_idx = np.array(sorted(row_union), dtype=int)
            M = np.zeros((T, len(rows_idx), N), dtype=np.float32)
            pos = {r: j for j, r in enumerate(rows_idx)}
            for t, d in enumerate(mats):
                for r, vec in d.items():
                    M[t, pos[r]] = vec
            arrays[k] = M
            arrays[f"{k}_rows"] = rows_idx

        saver = np.savez_compressed if cfg.compress_processed else np.savez
        saver(
            os.path.join(cfg.processed_dir, f"subnet_{netuid}.npz"),
            blocks=blocks, timestamps=stamps.astype(str), n_uids=n_uids,
            hotkeys=hot.astype(str), owner_hotkey=owner.astype(str),
            cert_count=cert_count, cert_flag=cert_flag,
            **{k: v.astype(str) for k, v in strings.items()}, **arrays)
        # identity: coldkeys / IPs collected with the snapshots, overridden by --identity if given
        ident = _identity_from_snapshots(hot, strings)
        for hk, d in identity.get(netuid, {}).items():
            ident.setdefault(hk, {}).update(d)
        if ident:
            with open(os.path.join(cfg.processed_dir, f"identity_{netuid}.json"), "w") as f:
                json.dump(ident, f)
        meta["netuids"].append(netuid)
        if verbose:
            print(f"  prepared netuid {netuid}: T={T}, N={N}, "
                  f"weight rows={len(arrays['weights_rows'])}")

    meta["layouts"] = {k: dict(v) for k, v in layouts.items()}
    meta["fields_present"] = sorted(entries[0].keys())
    with open(os.path.join(cfg.processed_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def _identity_from_snapshots(hot: np.ndarray, strings: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """Latest non-empty coldkey / axon IP seen for each hotkey."""
    out: Dict[str, dict] = {}
    ck, ip = strings.get("coldkeys"), strings.get("axon_ip")
    for t in range(hot.shape[0]):
        for u, hk in enumerate(hot[t]):
            if not hk:
                continue
            if ck is not None and ck[t, u]:
                out.setdefault(hk, {})["coldkey"] = ck[t, u]
            if ip is not None and ip[t, u] and ip[t, u] not in ("0.0.0.0", "None"):
                out.setdefault(hk, {})["ip"] = ip[t, u]
    return out


def _load_identity(path: str) -> Dict[int, Dict[str, dict]]:
    """Identity file: list of {"block_number", "netuid", "axons", "owners"}.

    `axons` is the raw Axons value for that subnet; `owners` maps hotkey -> coldkey.
    Returns {netuid: {hotkey: {"coldkey": ..., "ip": ...}}} merged over snapshots.
    """
    with open(path) as f:
        raw = json.load(f)
    out: Dict[int, Dict[str, dict]] = defaultdict(dict)
    for rec in raw:
        net = int(rec["netuid"])
        ips = parse_axons(rec.get("axons"))
        owners = {hk_to_str(k): hk_to_str(v) for k, v in (rec.get("owners") or {}).items()}
        for hk in set(ips) | set(owners):
            d = out[net].setdefault(hk, {})
            if owners.get(hk):
                d["coldkey"] = owners[hk]
            if ips.get(hk):
                d["ip"] = ips[hk]
    return dict(out)
