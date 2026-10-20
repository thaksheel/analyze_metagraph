"""BlockLoader: collector output -> one cached .npz of arrays per subnet.

Reads the collector's default fields straight from `exports/blocks/block_*.json.gz`
(streamed, one block at a time) or from one merged JSON list. Field names below are
the storage names the collector writes; edit FIELDS if yours differ.
"""
import glob
import gzip
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterator, List

import numpy as np

from .parsers import hk_to_str, parse_matrix, parse_scalar, parse_string_vector, parse_vector

# canonical name -> storage name in the collector output
VECTOR_FIELDS = {
    "incentive": "Incentive", "emission": "Emission", "dividends": "Dividends",
    "consensus": "Consensus", "vtrust": "ValidatorTrust", "active": "Active",
    "vpermit": "ValidatorPermit", "last_update": "LastUpdate", "stake_weight": "StakeWeight",
    "reg_block": "BlockAtRegistration",
}
SCALAR_FIELDS = {
    "tempo": "Tempo", "kappa": "Kappa", "rho": "Rho", "activity_cutoff": "ActivityCutoff",
    "mechanism": "SubnetMechanism", "mechanism_count": "MechanismCountCurrent",
    "n_uids_chain": "SubnetworkN", "subnet_tao": "SubnetTAO", "alpha_in": "SubnetAlphaIn",
    "alpha_out": "SubnetAlphaOut", "yc3": "Yuma3On", "commit_reveal": "CommitRevealWeightsEnabled",
    "immunity_period": "ImmunityPeriod", "max_validators": "MaxAllowedValidators", "burn_cost": "Burn",
}
HOTKEYS, OWNER, WEIGHTS = "Keys", "SubnetOwnerHotkey", "Weights"
EXACT = {"emission", "last_update", "reg_block"}   # kept as float64; other vectors float32
HYPERPARAMS = ["tempo", "kappa", "rho", "activity_cutoff", "mechanism", "mechanism_count",
               "immunity_period", "max_validators", "yc3", "commit_reveal"]


def iter_blocks(path: str) -> Iterator[dict]:
    """Yield block entries in block order from a directory of block files or a merged JSON."""
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "block_*.json.gz")) + glob.glob(os.path.join(path, "block_*.json"))
        def num(p):
            return int(os.path.basename(p).split("_")[1].split(".")[0])
        for p in sorted(files, key=num):
            opener = gzip.open if p.endswith(".gz") else open
            with opener(p, "rt") as f:
                yield json.load(f)
    else:
        with open(path) as f:
            entries = json.load(f)
        for e in sorted(entries, key=lambda e: int(e["block_info"]["number"])):
            yield e


class BlockLoader:
    def __init__(self, source: str, cache_dir: str):
        self.source = source
        self.cache_dir = cache_dir

    @property
    def meta_path(self) -> str:
        return os.path.join(self.cache_dir, "meta.json")

    def _signature(self) -> Dict:
        if os.path.isdir(self.source):
            files = glob.glob(os.path.join(self.source, "block_*.json*"))
            return {"source": os.path.abspath(self.source), "files": len(files),
                    "latest_mtime": max((os.path.getmtime(f) for f in files), default=0)}
        return {"source": os.path.abspath(self.source), "size": os.path.getsize(self.source),
                "mtime": os.path.getmtime(self.source)}

    def is_current(self) -> bool:
        if not os.path.exists(self.meta_path):
            return False
        with open(self.meta_path) as f:
            return json.load(f).get("signature") == self._signature()

    def build(self, force: bool = False, verbose: bool = True) -> Dict:
        if not force and self.is_current():
            with open(self.meta_path) as f:
                return json.load(f)
        os.makedirs(self.cache_dir, exist_ok=True)
        acc: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        layouts: Dict[str, Counter] = defaultdict(Counter)
        blocks, order, present, totals = [], [], set(), []
        for e in iter_blocks(self.source):
            nets = e["netuid"] if isinstance(e["netuid"], list) else [e["netuid"]]
            single = not isinstance(e["netuid"], list)
            if not order:
                order = [int(n) for n in nets]
            present |= set(e.keys())
            b = int(e["block_info"]["number"])
            blocks.append(b)
            totals.append(parse_scalar(e.get("TotalStake")))
            for i, net in enumerate(nets):
                def get(key):
                    v = e.get(key)
                    if single or v is None:
                        return v
                    return v[i] if isinstance(v, list) and i < len(v) else None
                a = acc[int(net)]
                a["blocks"].append(b)
                a["timestamps"].append(str(e["block_info"].get("timestamp", "")))
                for k, raw in VECTOR_FIELDS.items():
                    v, lay = parse_vector(get(raw))
                    a[k].append(v if k in EXACT else v.astype(np.float32))
                    layouts[k][lay] += 1
                for k, raw in SCALAR_FIELDS.items():
                    a[k].append(parse_scalar(get(raw)))
                hks, lay = parse_string_vector(get(HOTKEYS))
                a["hotkeys"].append(hks)
                layouts["hotkeys"][lay] += 1
                a["owner_hotkey"].append(hk_to_str(get(OWNER)) or "")
                # parse weights right away and keep only validator rows (uint16), so
                # memory stays at ~validators x uids per subnet-block
                nc = a["n_uids_chain"][-1]
                n_hint = max(len(a["incentive"][-1]), int(nc) if np.isfinite(nc) else 0)
                m, lay = parse_matrix(get(WEIGHTS), max(n_hint, 1))
                layouts["weights"][lay] += 1
                nz = np.where(m.sum(axis=1) > 0)[0]
                a["weights_rows_t"].append({int(r): m[r].astype(np.uint16) for r in nz})
        if not blocks:
            raise ValueError(f"no blocks found at {self.source}")

        netuids = []
        for net, a in sorted(acc.items()):
            T = len(a["blocks"])
            N = max([len(v) for v in a["incentive"]] + [int(x) for x in a["n_uids_chain"] if np.isfinite(x)] + [0])
            if N == 0:
                continue
            arrays = {"blocks": np.array(a["blocks"]), "timestamps": np.array(a["timestamps"]).astype(str),
                      "n_uids": np.array([len(v) for v in a["incentive"]])}
            for k in VECTOR_FIELDS:
                M = np.full((T, N), np.nan)
                for t, v in enumerate(a[k]):
                    M[t, :min(len(v), N)] = v[:N]
                arrays[k] = M
            for k in SCALAR_FIELDS:
                arrays[k] = np.array(a[k], dtype=float)
            hot = np.full((T, N), "", dtype=object)
            for t, hks in enumerate(a["hotkeys"]):
                for u, h in enumerate(hks[:N]):
                    hot[t, u] = h or ""
            arrays["hotkeys"] = hot.astype(str)
            arrays["owner_hotkey"] = np.array(a["owner_hotkey"]).astype(str)
            # weights: keep only rows that are ever non-zero (validators) -> [T, K, N]
            rows = np.array(sorted({r for d in a["weights_rows_t"] for r in d}), dtype=int)
            W = np.zeros((T, len(rows), N), dtype=np.float32)
            pos = {r: j for j, r in enumerate(rows)}
            for t, d in enumerate(a["weights_rows_t"]):
                for r, vec in d.items():
                    W[t, pos[r], :min(len(vec), N)] = vec[:N]
            a["weights_rows_t"] = None
            arrays["weights"] = W
            arrays["weights_rows"] = rows
            np.savez(os.path.join(self.cache_dir, f"subnet_{net}.npz"), **arrays)
            netuids.append(net)
            if verbose:
                print(f"  cached netuid {net}: {T} blocks, {N} uids, {len(rows)} validator rows", flush=True)
        meta = {"signature": self._signature(), "netuids": netuids, "netuid_order": order,
                "blocks": blocks, "total_stake": [None if not np.isfinite(x) else x for x in totals],
                "fields_present": sorted(present - {"block_info", "netuid", "_timings", "_errors"}),
                "layouts": {k: dict(v) for k, v in layouts.items()}}
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)
        return meta
