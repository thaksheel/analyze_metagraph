"""Storage items, snapshot container and file IO for the metagraph collector.

Every item is fetched with ONE `query_batch` call per block (like the version that
collected 500 blocks in 2 hours), using full storage keys:

  subnet       map netuid -> value                 params [netuid]            60 keys
  mechanism    map NetUidStorageIndex -> value     params [mechid*4096+netuid]
  uid          double map (netuid, uid)            params [netuid, uid]       one key per uid
  validator    double map (NetUidStorageIndex, uid) for validator uids only
               (Weights / Bonds: only permitted validators can set them)
  hotkey       double map (netuid, hotkey)         params [netuid, hotkey]
  hotkey_net   double map (hotkey, netuid)         params [hotkey, netuid]
  account      map hotkey -> value                 params [hotkey]
  global       StorageValue                        `query`, params []

Items are grouped so you switch groups on or off, and each item can be fetched every
N-th block only (`every`), which is how slow, slowly-changing items like Axons stay cheap.
"""
import glob
import gzip
import ipaddress
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

MECHANISM_STRIDE = 4096


@dataclass(frozen=True)
class Item:
    name: str                 # storage function
    kind: str                 # subnet | mechanism | uid | validator | hotkey | hotkey_net | account | global
    group: str                # used by `include=` to switch items on/off
    decode: str = "auto"      # auto | number | vector | account | pairs | axon_ip
    every: int = 1            # fetch every N-th block (forward-filled when merging)
    module: str = "SubtensorModule"
    out: str = ""             # field name in the output (default: name)

    @property
    def field(self) -> str:
        return self.out or self.name


ITEMS: List[Item] = [
    # --- core: the items your original collector fetched, one batch call each ---
    Item("Emission", "subnet", "core", "vector"),
    Item("Dividends", "subnet", "core", "vector"),
    Item("Active", "subnet", "core", "vector"),
    Item("ValidatorPermit", "subnet", "core", "vector"),
    Item("ValidatorTrust", "subnet", "core", "vector"),
    Item("Consensus", "subnet", "core", "vector"),
    Item("Incentive", "mechanism", "core", "vector"),
    Item("LastUpdate", "mechanism", "core", "vector"),
    Item("Tempo", "subnet", "core", "number"),
    Item("Kappa", "subnet", "core", "number"),
    Item("Rho", "subnet", "core", "number"),
    Item("ActivityCutoff", "subnet", "core", "number"),
    Item("SubnetMechanism", "subnet", "core", "number"),
    Item("MechanismCountCurrent", "subnet", "core", "number"),
    Item("SubnetworkN", "subnet", "core", "number"),
    # --- economics: cheap subnet-level values (one batch call each) ---
    Item("SubnetTAO", "subnet", "economics", "number"),
    Item("SubnetAlphaIn", "subnet", "economics", "number"),
    Item("SubnetAlphaOut", "subnet", "economics", "number"),
    Item("SubnetOwnerHotkey", "subnet", "economics", "account"),
    Item("Yuma3On", "subnet", "economics", "number"),
    Item("CommitRevealWeightsEnabled", "subnet", "economics", "number"),
    Item("StakeWeight", "subnet", "economics", "vector"),
    Item("ImmunityPeriod", "subnet", "economics", "number"),
    Item("MaxAllowedValidators", "subnet", "economics", "number"),
    Item("Burn", "subnet", "economics", "number"),
    # --- weights / bonds: one batch over validator uids ---
    Item("Weights", "validator", "weights", "pairs"),
    Item("Bonds", "validator", "bonds", "pairs"),
    # --- identity: one key per uid, small values ---
    Item("Keys", "uid", "keys", "account"),
    Item("BlockAtRegistration", "uid", "keys", "number"),
    # --- per-hotkey items (need Keys); larger, so sampled by default ---
    Item("TotalHotkeyAlpha", "hotkey_net", "stake", "number", every=1),
    Item("Owner", "account", "owner", "account", every=1, out="Coldkeys"),
    Item("Axons", "hotkey", "axons", "axon_ip", every=100, out="AxonIP"),
    Item("NeuronCertificates", "hotkey", "certificates", "auto", every=100),
    # --- network totals ---
    Item("TotalStake", "global", "economics", "number"),
]

DEFAULT_GROUPS = ["core", "economics", "weights", "keys"]
ALL_GROUPS = sorted({i.group for i in ITEMS})


class StorageFunctions(Enum):
    """For get_subnets_by(metric=...)."""
    emission = "Emission"
    dividends = "Dividends"
    active = "Active"
    incentive = "Incentive"
    subnet_tao = "SubnetTAO"


@dataclass
class BlockInfo:
    number: int
    hash: str
    timestamp: Optional[datetime] = None


# --------------------------------------------------------------------------- decoding

def plain(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return [plain(v) for v in x.tolist()]
    if isinstance(x, (bytes, bytearray)):
        return "0x" + bytes(x).hex()
    if isinstance(x, dict):
        return {str(plain(k)): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if hasattr(x, "value"):
        return plain(x.value)
    return str(x)


def as_number(x: Any) -> Optional[float]:
    x = plain(x)
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    if isinstance(x, dict):
        x = x.get("bits", x.get("value"))
    if isinstance(x, bool):
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def as_account(x: Any) -> Optional[str]:
    x = plain(x)
    if x is None or isinstance(x, str):
        return x
    while isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        x = x[0]
    if isinstance(x, list) and len(x) == 32 and all(isinstance(v, int) for v in x):
        raw = bytes(x)
        for mod, fn in (("scalecodec.utils.ss58", "ss58_encode"), ("bittensor.utils", "ss58_encode")):
            try:
                return getattr(__import__(mod, fromlist=[fn]), fn)(raw, 42)
            except Exception:
                pass
        return "0x" + raw.hex()
    return str(x)


def axon_ip(x: Any) -> Optional[str]:
    """Keep only the IP from AxonInfo (the rest of the struct is not stored)."""
    x = plain(x)
    if not isinstance(x, dict) or x.get("ip") in (None, 0, "0"):
        return None
    try:
        ip = int(x["ip"])
    except (TypeError, ValueError):
        return str(x["ip"])
    if int(as_number(x.get("ip_type")) or 4) == 6 or ip > 0xFFFFFFFF:
        return str(ipaddress.IPv6Address(ip))
    return str(ipaddress.IPv4Address(ip))


def decode(value: Any, how: str) -> Any:
    if how == "number":
        return as_number(value)
    if how == "account":
        return as_account(value)
    if how == "axon_ip":
        return axon_ip(value)
    if how == "vector":
        v = plain(value)
        v = v if isinstance(v, list) else ([] if v is None else [v])
        return [as_number(e) for e in v]
    if how == "pairs":
        return [[int(as_number(p[0])), int(as_number(p[1]))] for p in (plain(value) or [])]
    return plain(value)


# --------------------------------------------------------------------------- snapshot

@dataclass
class BlockSnapshot:
    """One block. `fields[name]` is a list aligned with `netuid` (as in your original
    BlockSnapshot), so `snap.Emission[i]` is subnet netuid[i]'s Emission vector.
    Weights / Bonds are stored sparse: [[src_uid, [[dest_uid, w_u16], ...]], ...];
    use `weights(netuid)` for the square [n_uids, n_uids] array."""
    block_info: BlockInfo
    netuid: List[int]
    fields: Dict[str, List[Any]] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def __getattr__(self, name):
        f = self.__dict__.get("fields", {})
        if name in f:
            return f[name]
        raise AttributeError(name)

    def _i(self, netuid: int) -> int:
        return self.netuid.index(netuid)

    def n_uids(self, netuid: int) -> int:
        i = self._i(netuid)
        n = (self.fields.get("SubnetworkN") or [None] * len(self.netuid))[i]
        if n:
            return int(n)
        inc = (self.fields.get("Incentive") or [[]] * len(self.netuid))[i]
        return len(inc or [])

    def weights(self, netuid: int, normalize: bool = False, name: str = "Weights") -> np.ndarray:
        """Square [n_uids, n_uids] array: rows = validator (setting) uid, cols = miner uid."""
        n = self.n_uids(netuid)
        M = np.zeros((n, n), dtype=np.float32 if normalize else np.uint16)
        rows = (self.fields.get(name) or [None] * len(self.netuid))[self._i(netuid)] or []
        for src, pairs in rows:
            for dest, w in pairs:
                if src < n and dest < n:
                    M[src, dest] = w
        if normalize:
            s = M.sum(axis=1, keepdims=True)
            M = np.divide(M, s, out=np.zeros_like(M), where=s > 0)
        return M

    def bonds(self, netuid: int) -> np.ndarray:
        return self.weights(netuid, name="Bonds")

    def validator_by_miner(self, netuid: int, normalize: bool = True):
        """(validator uids, [n_validators, n_uids]) rows of the square weight matrix."""
        vp = (self.fields.get("ValidatorPermit") or [None] * len(self.netuid))[self._i(netuid)]
        W = self.weights(netuid, normalize=normalize)
        v = np.where(np.asarray(vp, dtype=bool))[0] if vp else np.where(W.sum(axis=1) > 0)[0]
        return v, W[v]

    def to_dict(self) -> Dict:
        bi = self.block_info
        ts = bi.timestamp.isoformat() if isinstance(bi.timestamp, datetime) else bi.timestamp
        return {"block_info": {"number": int(bi.number), "hash": bi.hash, "timestamp": ts},
                "netuid": [int(n) for n in self.netuid], **self.fields,
                "_timings": self.timings, "_errors": self.errors}

    @classmethod
    def from_dict(cls, d: Dict) -> "BlockSnapshot":
        bi = d["block_info"]
        ts = bi.get("timestamp")
        info = BlockInfo(int(bi["number"]), bi["hash"], datetime.fromisoformat(ts) if ts else None)
        skip = {"block_info", "netuid", "_timings", "_errors"}
        return cls(info, [int(n) for n in d["netuid"]], {k: v for k, v in d.items() if k not in skip},
                   d.get("_timings", {}), d.get("_errors", []))


# --------------------------------------------------------------------------- files

def block_path(d: str, number: int) -> str:
    return os.path.join(d, f"block_{int(number)}.json.gz")


def collected_blocks(d: str) -> List[int]:
    out = []
    for p in glob.glob(os.path.join(d, "block_*.json.gz")):
        try:
            out.append(int(os.path.basename(p)[6:-8]))
        except ValueError:
            pass
    return sorted(out)


def save_block(snap: BlockSnapshot, d: str) -> str:
    os.makedirs(d, exist_ok=True)
    p = block_path(d, snap.block_info.number)
    with gzip.open(p + ".tmp", "wt", compresslevel=5) as f:
        json.dump(snap.to_dict(), f)
    os.replace(p + ".tmp", p)
    return p


def load_blocksnapshots(d: str, blocks: Optional[List[int]] = None) -> List[BlockSnapshot]:
    nums = collected_blocks(d)
    if blocks is not None:
        want = {int(b) for b in blocks}
        nums = [n for n in nums if n in want]
    out = []
    for n in nums:
        with gzip.open(block_path(d, n), "rt") as f:
            out.append(BlockSnapshot.from_dict(json.load(f)))
    return out


def merge_blocks(d: str, outpath: str, ffill: Optional[List[str]] = None) -> int:
    """Stream all block files into one JSON list (the layout bt_analysis reads).
    Fields fetched every N-th block are forward-filled from the last block that had them."""
    ffill = ffill if ffill is not None else [i.field for i in ITEMS if i.every > 1]
    last: Dict[str, Any] = {}
    nums = collected_blocks(d)
    with open(outpath, "w") as out:
        out.write("[")
        for k, n in enumerate(nums):
            with gzip.open(block_path(d, n), "rt") as f:
                e = json.load(f)
            for name in ffill:
                if name in e:
                    last[name] = (e["netuid"], e[name])
                elif name in last and last[name][0] == e["netuid"]:
                    e[name] = last[name][1]
            e.pop("_timings", None)
            e.pop("_errors", None)
            out.write(("," if k else "") + json.dumps(e))
        out.write("]")
    return len(nums)


def load_cache_block_info(bi_path: str) -> List[BlockInfo]:
    with open(bi_path) as f:
        data = json.load(f)
    return [BlockInfo(e["number"], e["hash"], datetime.fromisoformat(e["timestamp"])) for e in data]
