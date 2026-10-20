"""Storage layout, snapshot containers and value normalizers for the metagraph collector.

Each storage item comes from the runtime codegen (`storage_items.py`): the Item gives the
module, the storage name and the value type. `StorageSpec` adds what the descriptor does
not carry, the key layout, which decides how the item is queried:

  subnet         map NetUid -> value                       query_batch, params [netuid]
  mechanism      map NetUidStorageIndex -> value           query_batch, params [mechid*4096 + netuid]
  uid_matrix     double map (NetUidStorageIndex, uid) -> [(dest, u16)]
                                                           query_map, prefix [mechid*4096 + netuid]
  uid_vector     double map (NetUid, uid) -> value         query_map, prefix [netuid]
  hotkey_map     double map (NetUid, hotkey) -> value      query_map, prefix [netuid]; aligned to uid via Keys
  hotkey_netuid  double map (hotkey, NetUid) -> value      query_batch, params [hotkey, netuid] per uid
  owner          map hotkey -> coldkey                     query_batch, params [hotkey]; cached across blocks
  global         StorageValue                              query, params []
"""

import glob
import gzip
import ipaddress
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .storage_items import Commitments as CM
from .storage_items import SubtensorModule as SM

MECHANISM_STRIDE = 4096  # NetUidStorageIndex = mechid * 4096 + netuid

Kind = Literal[
    "subnet",
    "mechanism",
    "uid_matrix",
    "uid_vector",
    "hotkey_map",
    "hotkey_netuid",
    "owner",
    "global",
]
Shape = Literal["auto", "uid", "scalar", "raw"]


@dataclass(frozen=True)
class StorageSpec:
    item: Any  # codegen Item: (container, name, value_type_ident)
    kind: Kind
    optional: bool = (
        True  # optional items are skipped (and recorded) if their query fails
    )
    shape: Shape = (
        "auto"  # "uid": per-uid vector, "scalar": one value, "raw": keep as decoded
    )

    @property
    def name(self) -> str:
        return self.item.name

    @property
    def module(self) -> str:
        return self.item.container

    @property
    def ident(self) -> str:
        return getattr(self.item, "value_type_ident", "")


def S(item, kind: Kind, optional: bool = True, shape: Shape = "auto") -> StorageSpec:
    return StorageSpec(item, kind, optional, shape)


DEFAULT_SPECS: List[StorageSpec] = [
    # sizing (required; SubnetworkN sizes every array, MechanismCountCurrent lists mechanisms)
    S(SM.SubnetworkN, "subnet", optional=False, shape="scalar"),
    S(SM.MechanismCountCurrent, "subnet", shape="scalar"),
    S(SM.MechanismEmissionSplit, "subnet", shape="raw"),
    # per-uid vectors keyed by netuid
    S(SM.Emission, "subnet", optional=False, shape="uid"),
    S(SM.Dividends, "subnet", shape="uid"),
    S(SM.Active, "subnet", shape="uid"),
    S(SM.ValidatorPermit, "subnet", shape="uid"),
    S(SM.ValidatorTrust, "subnet", shape="uid"),
    S(SM.Consensus, "subnet", shape="uid"),
    S(SM.StakeWeight, "subnet", shape="uid"),
    # per-uid vectors keyed by NetUidStorageIndex (index == netuid for mechanism 0)
    S(SM.Incentive, "mechanism", optional=False, shape="uid"),
    S(SM.LastUpdate, "mechanism", shape="uid"),
    # subnet economics
    S(SM.SubnetTAO, "subnet", shape="scalar"),
    S(SM.SubnetAlphaIn, "subnet", shape="scalar"),
    S(SM.SubnetAlphaOut, "subnet", shape="scalar"),
    S(SM.SubnetVolume, "subnet", shape="scalar"),
    S(SM.SubnetMovingPrice, "subnet", shape="scalar"),
    S(SM.SubnetTaoInEmission, "subnet", shape="scalar"),
    S(SM.SubnetAlphaInEmission, "subnet", shape="scalar"),
    S(SM.SubnetAlphaOutEmission, "subnet", shape="scalar"),
    S(SM.SubnetOwner, "subnet", shape="scalar"),
    S(SM.SubnetOwnerHotkey, "subnet", shape="scalar"),
    S(SM.NetworkRegisteredAt, "subnet", shape="scalar"),
    # hyperparameters
    S(SM.SubnetMechanism, "subnet", shape="scalar"),
    S(SM.Tempo, "subnet", shape="scalar"),
    S(SM.Kappa, "subnet", shape="scalar"),
    S(SM.Rho, "subnet", shape="scalar"),
    S(SM.ActivityCutoff, "subnet", shape="scalar"),
    S(SM.ImmunityPeriod, "subnet", shape="scalar"),
    S(SM.MaxAllowedUids, "subnet", shape="scalar"),
    S(SM.MaxAllowedValidators, "subnet", shape="scalar"),
    S(SM.MinAllowedWeights, "subnet", shape="scalar"),
    S(SM.MaxWeightsLimit, "subnet", shape="scalar"),
    S(SM.WeightsVersionKey, "subnet", shape="scalar"),
    S(SM.WeightsSetRateLimit, "subnet", shape="scalar"),
    S(SM.BondsMovingAverage, "subnet", shape="scalar"),
    S(SM.BondsPenalty, "subnet", shape="scalar"),
    S(SM.CommitRevealWeightsEnabled, "subnet", shape="scalar"),
    S(SM.LiquidAlphaOn, "subnet", shape="scalar"),
    S(SM.Yuma3On, "subnet", shape="scalar"),
    S(SM.AlphaValues, "subnet", shape="raw"),
    S(SM.Burn, "subnet", shape="scalar"),
    S(SM.Difficulty, "subnet", shape="scalar"),
    S(SM.RegistrationsThisInterval, "subnet", shape="scalar"),
    S(SM.NetworkRegistrationAllowed, "subnet", shape="scalar"),
    # double maps
    S(SM.Weights, "uid_matrix", optional=False),
    S(SM.Bonds, "uid_matrix"),
    S(SM.Keys, "uid_vector", optional=False),
    S(SM.BlockAtRegistration, "uid_vector"),
    S(SM.Axons, "hotkey_map"),
    S(SM.Prometheus, "hotkey_map"),
    S(SM.NeuronCertificates, "hotkey_map"),
    S(CM.CommitmentOf, "hotkey_map"),
    # per-uid stake and delegation on this subnet
    S(SM.TotalHotkeyAlpha, "hotkey_netuid"),
    S(SM.ParentKeys, "hotkey_netuid"),
    # coldkey of each hotkey (queried once per new hotkey)
    S(SM.Owner, "owner"),
    # network-wide values
    S(SM.TotalStake, "global"),
    S(SM.TotalIssuance, "global"),
    S(SM.BlockEmission, "global"),
    S(SM.TaoWeight, "global"),
]


def validate_specs(specs: List[StorageSpec]) -> List[str]:
    """Names missing from the generated runtime module (empty when it isn't installed)."""
    from . import storage_items as si

    if si.GENERATED_SOURCE is None:
        return []
    missing = []
    for sp in specs:
        cls = getattr(si, sp.module, None)
        if cls is None or not hasattr(cls, sp.name):
            missing.append(f"{sp.module}.{sp.name}")
    return missing


class StorageFunctions(Enum):
    """Kept for `get_subnets_by(metric=...)` and backwards compatibility."""

    emission = "Emission"
    dividends = "Dividends"
    active = "Active"
    incentive = "Incentive"
    subnet_tao = "SubnetTAO"


@dataclass
class BlockInfo:
    number: int
    hash: str
    timestamp: datetime


@dataclass
class SubnetSnapshot:
    netuid: int
    n_uids: int
    mechanism_count: int = 1
    # per-uid vectors (length n_uids) and subnet scalars, by storage name
    fields: Dict[str, Any] = field(default_factory=dict)
    # mechid -> square [n_uids, n_uids] uint16; rows = setting (validator) uid, cols = miner uid
    weights: Dict[int, NDArray] = field(default_factory=dict)
    bonds: Dict[int, NDArray] = field(default_factory=dict)
    hotkeys: List[Optional[str]] = field(default_factory=list)  # index = uid
    coldkeys: List[Optional[str]] = field(
        default_factory=list
    )  # index = uid (from Owner)
    stake: Optional[NDArray] = None  # TotalHotkeyAlpha, alpha rao per uid
    # per-uid decoded records for hotkey-keyed items: Axons, Prometheus, NeuronCertificates,
    # CommitmentOf, ParentKeys; each a list of length n_uids (None where absent)
    records: Dict[str, List[Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)  # optional items that failed

    def axon_ips(self) -> List[Optional[str]]:
        return [
            a.get("ip") if isinstance(a, dict) else None
            for a in self.records.get("Axons", [None] * self.n_uids)
        ]

    def axon_blocks(self) -> NDArray:
        """Block of each uid's last serve_axon call (NaN if never served)."""
        return np.array(
            [
                float(a["block"]) if isinstance(a, dict) and a.get("block") else np.nan
                for a in self.records.get("Axons", [None] * self.n_uids)
            ]
        )

    def commitment_blocks(self) -> NDArray:
        """Block of each uid's latest Commitments.CommitmentOf entry (NaN if none)."""
        return np.array(
            [
                (
                    float(c["block"])
                    if isinstance(c, dict) and c.get("block") is not None
                    else np.nan
                )
                for c in self.records.get("CommitmentOf", [None] * self.n_uids)
            ]
        )

    def weight_matrix(self, mech: int = 0, normalize: bool = True) -> NDArray:
        """Square [n_uids, n_uids] weights; normalize=True gives row-stochastic float32."""
        w = self.weights.get(mech)
        if w is None:
            return np.zeros((self.n_uids, self.n_uids), dtype=np.float32)
        if not normalize:
            return w
        w = w.astype(np.float32)
        s = w.sum(axis=1, keepdims=True)
        return np.divide(w, s, out=np.zeros_like(w), where=s > 0)

    def validator_uids(self) -> NDArray:
        """UIDs with a validator permit (fallback: UIDs that set any weights)."""
        vp = self.fields.get("ValidatorPermit")
        if vp is not None and len(vp) == self.n_uids and np.asarray(vp).any():
            return np.where(np.asarray(vp, dtype=bool))[0]
        w = self.weights.get(0)
        return (
            np.where(w.sum(axis=1) > 0)[0] if w is not None else np.array([], dtype=int)
        )

    def validator_by_miner(self, mech: int = 0, normalize: bool = True):
        """(validator_uids, [n_validators, n_uids] weights) — the square matrix's validator rows."""
        v = self.validator_uids()
        return v, self.weight_matrix(mech, normalize)[v]


@dataclass
class BlockSnapshot:
    block_info: BlockInfo
    netuids: List[int]
    subnets: Dict[int, SubnetSnapshot] = field(default_factory=dict)
    globals: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# value normalizers (chain decoders return ints, tuples, dicts or wrapper objects)
# ---------------------------------------------------------------------------


def plain(x: Any) -> Any:
    """Convert decoded chain values to plain JSON-able Python."""
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
    if hasattr(x, "value"):  # ScaleType / ScaleObj style wrappers
        return plain(x.value)
    return str(x)


def as_int(x: Any) -> int:
    x = plain(x)
    while isinstance(x, list) and len(x) >= 1:
        x = x[-1]  # full storage keys come back as tuples; uid is last
    if isinstance(x, dict):
        x = next(iter(x.values()))
    return int(x)


def as_number(x: Any) -> float:
    x = plain(x)
    if isinstance(x, list):
        x = x[0] if len(x) == 1 else np.nan
    if isinstance(x, dict):
        for k in ("value", "bits"):
            if k in x:
                return as_number(x[k])
        return np.nan
    if isinstance(x, bool):
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _ss58(raw: bytes) -> Optional[str]:
    for mod, fn in (
        ("scalecodec.utils.ss58", "ss58_encode"),
        ("bittensor.utils", "ss58_encode"),
    ):
        try:
            m = __import__(mod, fromlist=[fn])
            return getattr(m, fn)(raw, 42)
        except Exception:
            continue
    return None


def as_account(x: Any) -> Optional[str]:
    """AccountId -> ss58 string when possible, else 0x-hex."""
    x = plain(x)
    if x is None:
        return None
    if isinstance(x, str):
        return x
    while isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        x = x[0]
    if isinstance(x, list) and len(x) == 32 and all(isinstance(v, int) for v in x):
        raw = bytes(x)
        return _ss58(raw) or "0x" + raw.hex()
    if isinstance(x, dict):
        for k in ("ss58", "address", "value"):
            if k in x:
                return as_account(x[k])
    return str(x)


def to_vector(raw: Any, n: int, dtype=np.float64) -> NDArray:
    """Subnet vector (one value per uid) -> length-n array, padded/trimmed."""
    raw = plain(raw)
    out = np.zeros(n, dtype=dtype)
    if raw is None:
        return out
    if not isinstance(raw, list):
        raw = [raw]
    vals = np.array([as_number(v) for v in raw], dtype=float)
    k = min(n, len(vals))
    out[:k] = np.nan_to_num(vals[:k])
    return out


# ---------------------------------------------------------------------------
# decoding by value_type_ident
# ---------------------------------------------------------------------------

# substrate_fixed types: the runtime stores SubnetMovingPrice as I96F32 (32 fractional bits)
FIXED_FRACTION_BITS = {"SubnetMovingPrice": 32}


def ip_to_str(ip: Any, ip_type: Any = None) -> Optional[str]:
    try:
        ip = int(as_number(ip))
    except (TypeError, ValueError, OverflowError):
        return None
    if ip <= 0:
        return None
    t = as_number(ip_type) if ip_type is not None else np.nan
    if t == 6 or ip > 0xFFFFFFFF:
        return str(ipaddress.IPv6Address(ip))
    return str(ipaddress.IPv4Address(ip))


def _axon(v: Any) -> Optional[Dict]:
    v = plain(v)
    if not isinstance(v, dict):
        return None
    out = {
        k: as_number(v.get(k))
        for k in ("block", "version", "port", "ip_type", "protocol")
        if k in v
    }
    out["ip"] = ip_to_str(v.get("ip"), v.get("ip_type"))
    return out


def _registration(v: Any) -> Optional[Dict]:
    v = plain(v)
    if not isinstance(v, dict):
        return None
    info = v.get("info") or {}
    fields_ = info.get("fields", []) if isinstance(info, dict) else []
    return {
        "block": as_number(v.get("block")),
        "deposit": as_number(v.get("deposit")),
        "n_fields": len(fields_) if isinstance(fields_, list) else np.nan,
    }


def _fixed(v: Any, name: str) -> float:
    v = plain(v)
    if isinstance(v, dict) and "bits" in v:
        return float(int(v["bits"])) / 2 ** FIXED_FRACTION_BITS.get(name, 32)
    return as_number(v)


def decode(value: Any, spec: "StorageSpec") -> Any:
    """Decode one storage value using the runtime's value_type_ident."""
    t = spec.ident
    if t == "AccountId32":
        return as_account(value)
    if t in ("AxonInfo", "PrometheusInfo"):
        return _axon(value)
    if t == "Registration":
        return _registration(value)
    if t.startswith("Fixed"):
        return _fixed(value, spec.name)
    if t == "Vec<(u64, AccountId32)>":
        return [
            [as_number(p[0]), as_account(p[1])]
            for p in (plain(value) or [])
            if isinstance(p, list) and len(p) == 2
        ]
    if t in (
        "bool",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "MechId",
        "TaoBalance",
        "AlphaBalance",
        "PerU16",
    ):
        return as_number(value)
    return plain(value)


# ---------------------------------------------------------------------------
# JSON (de)serialization; matrices are stored sparse: [[src, [[dest, w], ...]], ...]
# ---------------------------------------------------------------------------


def matrix_to_sparse(m: NDArray) -> List:
    rows = []
    for src in np.where(m.any(axis=1))[0]:
        dest = np.nonzero(m[src])[0]
        rows.append([int(src), [[int(d), int(m[src, d])] for d in dest]])
    return rows


def sparse_to_matrix(rows: List, n: int, dtype=np.uint16) -> NDArray:
    m = np.zeros((n, n), dtype=dtype)
    for src, pairs in rows:
        for dest, w in pairs:
            if src < n and dest < n:
                m[src, dest] = w
    return m


def snapshot_to_dict(s: BlockSnapshot) -> Dict:
    bi = s.block_info
    ts = (
        bi.timestamp.isoformat() if isinstance(bi.timestamp, datetime) else bi.timestamp
    )
    out = {
        "block_info": {"number": int(bi.number), "hash": bi.hash, "timestamp": ts},
        "netuids": list(map(int, s.netuids)),
        "globals": plain(s.globals),
        "subnets": {},
    }
    for net, sn in s.subnets.items():
        out["subnets"][str(net)] = {
            "netuid": sn.netuid,
            "n_uids": sn.n_uids,
            "mechanism_count": sn.mechanism_count,
            "fields": {k: plain(v) for k, v in sn.fields.items()},
            "weights": {str(k): matrix_to_sparse(v) for k, v in sn.weights.items()},
            "bonds": {str(k): matrix_to_sparse(v) for k, v in sn.bonds.items()},
            "hotkeys": sn.hotkeys,
            "stake": plain(sn.stake) if sn.stake is not None else None,
            "coldkeys": sn.coldkeys,
            "records": {k: [plain(x) for x in v] for k, v in sn.records.items()},
            "errors": sn.errors,
        }
    return out


def snapshot_from_dict(d: Dict, netuids: Optional[List[int]] = None) -> BlockSnapshot:
    bi = d["block_info"]
    snap = BlockSnapshot(
        block_info=BlockInfo(
            number=int(bi["number"]),
            hash=bi["hash"],
            timestamp=(
                datetime.fromisoformat(bi["timestamp"]) if bi.get("timestamp") else None
            ),
        ),
        netuids=[int(n) for n in d["netuids"]],
        globals=d.get("globals", {}),
    )
    for key, sd in d["subnets"].items():
        net = int(key)
        if netuids is not None and net not in netuids:
            continue
        n = int(sd["n_uids"])
        fields = {
            k: (np.asarray(v) if isinstance(v, list) else v)
            for k, v in sd["fields"].items()
        }
        snap.subnets[net] = SubnetSnapshot(
            netuid=net,
            n_uids=n,
            mechanism_count=int(sd.get("mechanism_count", 1)),
            fields=fields,
            weights={
                int(k): sparse_to_matrix(v, n) for k, v in sd.get("weights", {}).items()
            },
            bonds={
                int(k): sparse_to_matrix(v, n) for k, v in sd.get("bonds", {}).items()
            },
            hotkeys=sd.get("hotkeys", []),
            stake=(
                np.asarray(sd["stake"], dtype=float)
                if sd.get("stake") is not None
                else None
            ),
            coldkeys=sd.get("coldkeys", []),
            records=sd.get("records", {}),
            errors=sd.get("errors", []),
        )
    return snap


def load_cache_block_info(bi_path: str) -> List[BlockInfo]:
    """Collects only the hash, number, and timestamps as `BlockInfo` fields while ignoring the rest."""
    with open(bi_path, "r") as f:
        data = json.load(f)
    return [
        BlockInfo(
            number=e["number"],
            hash=e["hash"],
            timestamp=datetime.fromisoformat(e["timestamp"]),
        )
        for e in data
    ]


def collected_blocks(indir: str) -> List[int]:
    out = []
    for p in glob.glob(os.path.join(indir, "block_*.json.gz")):
        try:
            out.append(int(os.path.basename(p)[6:-8]))
        except ValueError:
            pass
    return sorted(out)


def load_blocksnapshots(
    indir: str, blocks: Optional[List[int]] = None, netuids: Optional[List[int]] = None
) -> List[BlockSnapshot]:
    """Load per-block files written by the collector (optionally a subset of blocks / subnets).
    Weights and Bonds come back as square [n_uids, n_uids] uint16 arrays per mechanism.
    """
    nums = collected_blocks(indir)
    if blocks is not None:
        want = {int(b) for b in blocks}
        nums = [n for n in nums if n in want]
    out = []
    for n in nums:
        with gzip.open(os.path.join(indir, f"block_{n}.json.gz"), "rt") as f:
            out.append(snapshot_from_dict(json.load(f), netuids=netuids))
    return out
