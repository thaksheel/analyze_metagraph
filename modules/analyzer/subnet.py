"""Per-subnet data access with derived quantities used by every analysis."""
import json
import os
from functools import cached_property
from typing import Dict, Iterator, List, Optional

import numpy as np

from .config import RAO_PER_TAO, SECONDS_PER_BLOCK, Config


def row_share(a: np.ndarray) -> np.ndarray:
    """Normalize each row to sum 1, NaN-safe; rows summing to 0 stay 0."""
    a = np.nan_to_num(a, nan=0.0)
    s = a.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(s > 0, a / s, 0.0)
    return out


class SubnetData:
    def __init__(self, netuid: int, path: str, identity_path: Optional[str] = None):
        self.netuid = netuid
        z = np.load(path, allow_pickle=False)
        self._z = {k: z[k] for k in z.files}
        self.identity: Dict[str, dict] = {}
        if identity_path and os.path.exists(identity_path):
            with open(identity_path) as f:
                self.identity = json.load(f)

    def __getattr__(self, k):
        z = self.__dict__.get("_z", {})
        if k in z:
            return z[k]
        raise AttributeError(k)

    # ---- shape / time ----
    @property
    def T(self) -> int:
        return len(self.blocks)

    @property
    def N(self) -> int:
        return self.incentive.shape[1]

    @cached_property
    def hours(self) -> np.ndarray:
        return (self.blocks - self.blocks[0]) * SECONDS_PER_BLOCK / 3600.0

    @cached_property
    def spacing_blocks(self) -> float:
        return float(np.median(np.diff(self.blocks))) if self.T > 1 else 1.0

    @cached_property
    def interval_blocks(self) -> np.ndarray:
        """Blocks represented by each snapshot (gap to the next one)."""
        d = np.diff(self.blocks).astype(float)
        return np.append(d, d[-1] if len(d) else 1.0)

    def snaps(self, days: float) -> int:
        return max(1, int(round(days * 86400 / (self.spacing_blocks * SECONDS_PER_BLOCK))))

    # ---- masks ----
    @cached_property
    def exists(self) -> np.ndarray:
        return ~np.isnan(self.incentive)

    @cached_property
    def is_validator(self) -> np.ndarray:
        return np.nan_to_num(self.vpermit) > 0

    @cached_property
    def is_miner(self) -> np.ndarray:
        return self.exists & ~self.is_validator

    # ---- shares ----
    @cached_property
    def inc_share(self) -> np.ndarray:
        return row_share(self.incentive)

    @cached_property
    def emi_share(self) -> np.ndarray:
        return row_share(self.emission)

    @cached_property
    def div_share(self) -> np.ndarray:
        return row_share(self.dividends)

    @cached_property
    def miner_emi_share(self) -> np.ndarray:
        """Each miner's share of emission paid to miners (incentive side)."""
        return row_share(np.where(self.is_miner, np.nan_to_num(self.incentive), 0.0))

    # ---- value ----
    @cached_property
    def tempo_v(self) -> np.ndarray:
        t = np.array(self.tempo, dtype=float)
        return np.where(np.isfinite(t) & (t > 0), t, 360.0)

    @cached_property
    def price(self) -> np.ndarray:
        """TAO per alpha (NaN if SubnetTAO / SubnetAlphaIn not collected)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.array(self.subnet_tao, dtype=float) / np.array(self.alpha_in, dtype=float)
        return np.where(np.isfinite(p) & (p > 0), p, np.nan)

    @property
    def has_price(self) -> bool:
        return bool(np.isfinite(self.price).any())

    @cached_property
    def alpha_flow(self) -> np.ndarray:
        """Alpha emitted to each UID over the interval each snapshot represents.
        Assumes `Emission` holds the per-UID amount (rao) of the last epoch."""
        epochs = self.interval_blocks / (self.tempo_v + 1.0)
        return np.nan_to_num(self.emission) / RAO_PER_TAO * epochs[:, None]

    @cached_property
    def tao_flow(self) -> np.ndarray:
        return self.alpha_flow * np.nan_to_num(self.price)[:, None]

    @cached_property
    def subnet_tao_flow(self) -> np.ndarray:
        return self.tao_flow.sum(axis=1)

    # ---- weights ----
    @cached_property
    def W(self) -> np.ndarray:
        """[T, K, N] row-normalized weights from rows `weights_rows`."""
        return row_share(self.weights).astype(np.float32)

    @cached_property
    def B(self) -> np.ndarray:
        return row_share(self.bonds).astype(np.float32)

    def active_validator_rows(self, t: int) -> np.ndarray:
        """Indices into W rows for validators with permit and a non-empty row at t."""
        rows = self.weights_rows
        if len(rows) == 0:
            return np.array([], dtype=int)
        has = self.W[t].sum(axis=1) > 0
        permit = self.is_validator[t, rows] if self.N else np.zeros(len(rows), bool)
        return np.where(has & permit)[0]

    # ---- identity ----
    @property
    def has_hotkeys(self) -> bool:
        return bool((self.hotkeys != "").any())

    @property
    def has_reg_block(self) -> bool:
        return bool(np.isfinite(self.reg_block).any())

    @cached_property
    def replaced(self) -> np.ndarray:
        """[T, N] True where a UID got a new occupant since the previous snapshot.
        Uses Keys / BlockAtRegistration; falls back to a Bonds-column reset."""
        r = np.zeros((self.T, self.N), dtype=bool)
        if self.has_hotkeys:
            h = self.hotkeys
            r[1:] |= (h[1:] != h[:-1]) & (h[1:] != "") & (h[:-1] != "")
        if self.has_reg_block:
            rb = self.reg_block
            with np.errstate(invalid="ignore"):
                r[1:] |= np.isfinite(rb[1:]) & np.isfinite(rb[:-1]) & (rb[1:] != rb[:-1])
        if not (self.has_hotkeys or self.has_reg_block) and self.bonds.shape[1] > 0:
            col = self.bonds.sum(axis=1)  # [T, N] bonds received
            r[1:] |= (col[:-1] > 0) & (col[1:] == 0)
        return r

    @cached_property
    def occupant(self) -> np.ndarray:
        """[T, N] stable identity string per UID slot: hotkey if known, else uid:epoch."""
        if self.has_hotkeys:
            return self.hotkeys
        epoch = np.cumsum(self.replaced, axis=0)
        uid = np.broadcast_to(np.arange(self.N), (self.T, self.N))
        return np.char.add(np.char.add(uid.astype(str), ":"), epoch.astype(str))

    @cached_property
    def owner_uid(self) -> np.ndarray:
        """UID owned by SubnetOwnerHotkey at each t (falls back to UID 0)."""
        out = np.zeros(self.T, dtype=int)
        if (self.owner_hotkey != "").any() and self.has_hotkeys:
            for t in range(self.T):
                m = np.where(self.hotkeys[t] == self.owner_hotkey[t])[0]
                out[t] = m[0] if len(m) else 0
        return out

    def coldkey(self, hk: str) -> Optional[str]:
        return self.identity.get(hk, {}).get("coldkey")

    def ip(self, hk: str) -> Optional[str]:
        return self.identity.get(hk, {}).get("ip")


class Dataset:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        with open(os.path.join(cfg.processed_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.netuids: List[int] = self.meta["netuids"]

    def load(self, netuid: int) -> SubnetData:
        d = self.cfg.processed_dir
        return SubnetData(netuid, os.path.join(d, f"subnet_{netuid}.npz"),
                          os.path.join(d, f"identity_{netuid}.json"))

    def __iter__(self) -> Iterator[SubnetData]:
        for n in self.netuids:
            yield self.load(n)
