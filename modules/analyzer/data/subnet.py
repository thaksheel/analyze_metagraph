"""Subnet: every array and derived quantity for one subnet, in one place.

Raw arrays ([T] per block or [T, N] per block and uid) are attributes named as in
loader.VECTOR_FIELDS / SCALAR_FIELDS (incentive, emission, consensus, stake_weight,
reg_block, tempo, subnet_tao, ...). Derived quantities are cached properties:

  time        T, N, blocks, hours, snaps(days)
  masks       exists, is_validator, is_miner
  shares      inc_share, emi_share, div_share, miner_emi_share
  value       price (TAO per alpha), alpha_flow, tao_flow, subnet_tao_flow
  weights     W [T, K, N] row-normalized, weights_rows, active_validator_rows(t),
              validator_weight(t, uids)  (StakeWeight, falling back to ValidatorTrust)
  identity    replaced, occupant, owner_uid
  support     median / mean validator support, breadth, CV, consensus share per miner
"""
import os
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Tuple

import numpy as np

from ..config import RAO, SECONDS_PER_BLOCK

EPS = 1e-6


def row_share(a: np.ndarray) -> np.ndarray:
    """Normalize each row to sum 1 (NaN-safe); rows summing to 0 stay 0."""
    a = np.nan_to_num(np.asarray(a, dtype=float), nan=0.0)
    s = a.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, a / s, 0.0)


@dataclass
class Support:
    median_share: np.ndarray   # median weight across validators (one validator, one vote)
    mean_share: np.ndarray
    breadth: np.ndarray        # share of validators giving >= a fair share (1 / n_miners)
    cv: np.ndarray             # coefficient of variation of received weight
    cons_share: np.ndarray     # Consensus storage, renormalized over miners


class Subnet:
    def __init__(self, netuid: int, path: str):
        self.netuid = netuid
        with np.load(path, allow_pickle=False) as z:
            self._z = {k: z[k] for k in z.files}

    def __getattr__(self, k):
        z = self.__dict__.get("_z", {})
        if k in z:
            return z[k]
        raise AttributeError(f"Subnet has no field '{k}'")

    def __repr__(self):
        return f"Subnet(netuid={self.netuid}, blocks={self.T}, uids={self.N}, validators={len(self.weights_rows)})"

    # ---------------------------------------------------------------- time
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
        d = np.diff(self.blocks).astype(float)
        return np.append(d, d[-1] if len(d) else 1.0)

    @property
    def period_days(self) -> float:
        return float((self.blocks[-1] - self.blocks[0]) * SECONDS_PER_BLOCK / 86400)

    def snaps(self, days: float) -> int:
        """Number of snapshots covering `days`."""
        return max(1, int(round(days * 86400 / (self.spacing_blocks * SECONDS_PER_BLOCK))))

    # ---------------------------------------------------------------- masks
    @cached_property
    def exists(self) -> np.ndarray:
        return ~np.isnan(self.incentive)

    @cached_property
    def is_validator(self) -> np.ndarray:
        return np.nan_to_num(self.vpermit) > 0

    @cached_property
    def is_miner(self) -> np.ndarray:
        return self.exists & ~self.is_validator

    # ---------------------------------------------------------------- shares
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
        """Each miner's share of the emission paid to miners."""
        return row_share(np.where(self.is_miner, np.nan_to_num(self.incentive), 0.0))

    # ---------------------------------------------------------------- value
    @cached_property
    def tempo_v(self) -> np.ndarray:
        t = np.array(self.tempo, dtype=float)
        return np.where(np.isfinite(t) & (t > 0), t, 360.0)

    @cached_property
    def price(self) -> np.ndarray:
        """TAO per alpha = SubnetTAO / SubnetAlphaIn (NaN if missing)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            p = self.subnet_tao / self.alpha_in
        return np.where(np.isfinite(p) & (p > 0), p, np.nan)

    @property
    def has_price(self) -> bool:
        return bool(np.isfinite(self.price).any())

    @cached_property
    def alpha_flow(self) -> np.ndarray:
        """Alpha paid to each uid over the interval a snapshot stands for
        (Emission = last epoch's amount in rao; interval / (tempo + 1) epochs)."""
        epochs = self.interval_blocks / (self.tempo_v + 1.0)
        return np.nan_to_num(self.emission) / RAO * epochs[:, None]

    @cached_property
    def tao_flow(self) -> np.ndarray:
        return self.alpha_flow * np.nan_to_num(self.price)[:, None]

    @cached_property
    def subnet_tao_flow(self) -> np.ndarray:
        return self.tao_flow.sum(axis=1)

    def miner_flow(self, unit: str = "alpha") -> np.ndarray:
        f = self.alpha_flow if unit == "alpha" else self.tao_flow
        return np.where(self.is_miner, f, 0).sum(axis=1)

    # ---------------------------------------------------------------- weights
    @cached_property
    def W(self) -> np.ndarray:
        """[T, K, N] row-normalized weights for rows `weights_rows` (validator uids)."""
        return row_share(self.weights).astype(np.float32)

    def active_validator_rows(self, t: int) -> np.ndarray:
        """Indices into W rows: validators with a permit and a non-empty row at t."""
        rows = self.weights_rows
        if len(rows) == 0:
            return np.array([], dtype=int)
        return np.where((self.W[t].sum(axis=1) > 0) & self.is_validator[t, rows])[0]

    @property
    def has_stake_weight(self) -> bool:
        return bool(np.nan_to_num(self.stake_weight).sum() > 0)

    def validator_weight(self, t: int, uids: np.ndarray) -> np.ndarray:
        """Influence of each validator: StakeWeight, or ValidatorTrust if StakeWeight is empty."""
        sw = np.nan_to_num(self.stake_weight[t, uids])
        return sw if sw.sum() > 0 else np.nan_to_num(self.vtrust[t, uids])

    # ---------------------------------------------------------------- identity
    @property
    def has_hotkeys(self) -> bool:
        return bool((self.hotkeys != "").any())

    @property
    def has_reg_block(self) -> bool:
        return bool(np.isfinite(self.reg_block).any())

    @cached_property
    def replaced(self) -> np.ndarray:
        """[T, N] True where a uid has a new occupant (hotkey or registration block changed)."""
        r = np.zeros((self.T, self.N), dtype=bool)
        if self.has_hotkeys:
            h = self.hotkeys
            r[1:] |= (h[1:] != h[:-1]) & (h[1:] != "") & (h[:-1] != "")
        if self.has_reg_block:
            rb = self.reg_block
            with np.errstate(invalid="ignore"):
                r[1:] |= np.isfinite(rb[1:]) & np.isfinite(rb[:-1]) & (rb[1:] != rb[:-1])
        return r

    @cached_property
    def occupant(self) -> np.ndarray:
        """[T, N] stable identity per uid slot: the hotkey, else 'uid:registration-count'."""
        if self.has_hotkeys:
            return self.hotkeys
        epoch = np.cumsum(self.replaced, axis=0)
        uid = np.broadcast_to(np.arange(self.N), (self.T, self.N))
        return np.char.add(np.char.add(uid.astype(str), ":"), epoch.astype(str))

    @cached_property
    def owner_uid(self) -> np.ndarray:
        """uid held by SubnetOwnerHotkey at each block (falls back to uid 0)."""
        out = np.zeros(self.T, dtype=int)
        if (self.owner_hotkey != "").any() and self.has_hotkeys:
            for t in range(self.T):
                m = np.where(self.hotkeys[t] == self.owner_hotkey[t])[0]
                out[t] = m[0] if len(m) else 0
        return out

    # ---------------------------------------------------------------- validator support
    @cached_property
    def support(self) -> Support:
        T, N = self.T, self.N
        med, mean, breadth, cv = (np.zeros((T, N)) for _ in range(4))
        for t in range(T):
            miners = self.is_miner[t]
            vr = self.active_validator_rows(t)
            if not len(vr) or not miners.any():
                continue
            Wm = self.W[t][vr][:, miners]
            Wm = Wm / np.maximum(Wm.sum(axis=1, keepdims=True), 1e-12)
            m = Wm.mean(axis=0)
            med[t, miners] = np.median(Wm, axis=0)
            mean[t, miners] = m
            breadth[t, miners] = (Wm >= 1.0 / miners.sum()).mean(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                cv[t, miners] = np.where(m > 0, Wm.std(axis=0) / m, 0.0)
        cons = np.where(self.is_miner, np.nan_to_num(self.consensus), 0.0)
        return Support(row_share(med), row_share(mean), breadth, cv, row_share(cons))

    # ---------------------------------------------------------------- behavior profiles
    def miner_profiles(self, a: int, b: int, uids: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """(trajectory [n, w], validator-preference profile [n, K] or None) over [a, b).
        Trajectory = log share minus the cross-sectional median (removes common shocks)."""
        ls = np.log(self.miner_emi_share[a:b] + EPS)
        miners = self.is_miner[a:b]
        med = np.array([np.median(ls[t][miners[t]]) if miners[t].any() else 0.0 for t in range(b - a)])
        traj = (ls[:, uids] - med[:, None]).T
        prof = None
        if len(self.weights_rows):
            Wv = self.W[a:b][:, :, uids]
            used = Wv.sum(axis=(0, 2)) > 0
            if used.sum() >= 2:
                lw = np.log(Wv[:, used, :] + EPS)
                prof = (lw - lw.mean(axis=1, keepdims=True)).mean(axis=0).T
        return traj, prof
