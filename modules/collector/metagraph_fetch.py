import asyncio
import gzip
import inspect
import json
import os
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from tqdm import tqdm

from .utils import (
    DEFAULT_SPECS,
    MECHANISM_STRIDE,
    BlockInfo,
    BlockSnapshot,
    StorageFunctions,
    StorageSpec,
    SubnetSnapshot,
    as_account,
    as_int,
    as_number,
    decode,
    plain,
    snapshot_to_dict,
    to_vector,
    validate_specs,
)
from .utils import collected_blocks as _collected_blocks
from .utils import load_blocksnapshots as _load_blocksnapshots

try:
    from bittensor.result import RpcPolicyError
except Exception: 
    class RpcPolicyError(Exception):
        pass


MODULE = "SubtensorModule"
# errors that will not go away by retrying (bad storage name, bad params, decoding bugs)
NON_RETRYABLE = (TypeError, KeyError, ValueError, AttributeError, NotImplementedError)


class RetryExhausted(RuntimeError):
    pass


def mech_index(netuid: int, mechid: int) -> int:
    return mechid * MECHANISM_STRIDE + netuid


async def _collect(res: Any) -> List:
    """query_map results may be a list, an iterable or an async iterator."""
    if res is None:
        return []
    if hasattr(res, "__aiter__"):
        return [item async for item in res]
    if isinstance(res, (list, tuple)):
        return list(res)
    if hasattr(res, "records"):
        return list(res.records)
    return list(res)


def _kv(item: Any) -> Tuple[Any, Any]:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return item[0], item[1]
    if hasattr(item, "key") and hasattr(item, "value"):
        return item.key, item.value
    raise ValueError(f"Unexpected query_map item: {item!r}")


class ChainIO:
    """Thin wrapper over the substrate interface: retries, rate limiting, call shapes."""

    def __init__(self, substrate: Any, manager: "MetagraphManager"):
        self.subs = substrate
        self.m = manager
        self.sem = asyncio.Semaphore(manager.max_concurrency)

    async def _call(self, fn: Callable, what: str):
        async def guarded():
            async with self.sem:
                r = fn()
                return await r if inspect.isawaitable(r) else r

        return await self.m.historical_retry(guarded, what=what)

    @staticmethod
    def _ms(storage: Any) -> Tuple[str, str]:
        """Accept a codegen Item, a StorageSpec or a plain SubtensorModule storage name."""
        if isinstance(storage, StorageSpec):
            return storage.module, storage.name
        if hasattr(storage, "container") and hasattr(storage, "name"):
            return storage.container, storage.name
        return MODULE, str(storage)

    async def query_map(
        self, storage: Any, params: Sequence, block_hash: str
    ) -> List[Tuple[Any, Any]]:
        mod, name = self._ms(storage)
        res = await self._call(
            lambda: self.subs.query_map(mod, name, list(params), block_hash=block_hash),
            what=f"query_map {mod}.{name}{list(params)}",
        )
        return [_kv(it) for it in await _collect(res)]

    async def query(self, storage: Any, params: Sequence, block_hash: str) -> Any:
        mod, name = self._ms(storage)
        return await self._call(
            lambda: self.subs.query(mod, name, list(params), block_hash=block_hash),
            what=f"query {mod}.{name}{list(params)}",
        )

    async def query_batch(
        self, storage: Any, param_sets: List[List], block_hash: str
    ) -> List[Any]:
        mod, name = self._ms(storage)
        if not param_sets:
            return []
        if hasattr(self.subs, "query_batch"):
            return list(
                await self._call(
                    lambda: self.subs.query_batch(
                        module=mod,
                        storage_function=name,
                        block_hash=block_hash,
                        param_sets=param_sets,
                    ),
                    what=f"query_batch {mod}.{name} x{len(param_sets)}",
                )
            )
        return list(
            await asyncio.gather(
                *(self.query(storage, p, block_hash) for p in param_sets)
            )
        )


class MetagraphManager:
    def __init__(
        self,
        display: bool = False,
        network: str = "finney",
        client_kwargs: Optional[Dict] = None,
        specs: Optional[List[StorageSpec]] = None,
        max_concurrency: int = 4,
        initial_wait: float = 5.0,
        max_wait: float = 600.0,
        max_attempts: int = 8,
        max_reconnects: int = 20,
        client_factory: Optional[Callable] = None,
    ):
        """
        client_kwargs   passed to `bittensor.Subtensor(network, **client_kwargs)`
        specs           storage items to collect (default: utils.DEFAULT_SPECS)
        client_factory  zero-arg callable returning an async context manager whose value
                        has `._substrate`; used for tests or a custom client
        """
        self.display = display
        self.network = network
        self.client_kwargs = client_kwargs or {}
        self.specs = list(specs or DEFAULT_SPECS)
        self.max_concurrency = max_concurrency
        self.initial_wait = initial_wait
        self.max_wait = max_wait
        self.max_attempts = max_attempts
        self.max_reconnects = max_reconnects
        self.client_factory = client_factory
        self.owner_cache: Dict[str, Optional[str]] = (
            {}
        )  # hotkey -> coldkey, reused across blocks
        missing = validate_specs(self.specs)
        if missing:
            print(f"[specs] not in the generated runtime metadata, skipped: {missing}")
            self.specs = [
                sp for sp in self.specs if f"{sp.module}.{sp.name}" not in missing
            ]

    # ------------------------------------------------------------------ connection
    def _open_client(self):
        if self.client_factory is not None:
            return self.client_factory()
        import bittensor as bt

        return bt.Subtensor(self.network, **self.client_kwargs)

    def _log(self, msg: str):
        if self.display:
            print(msg)

    async def historical_retry(
        self,
        coro_factory: Callable,
        initial_wait: Optional[float] = None,
        max_wait: Optional[float] = None,
        max_attempts: Optional[int] = None,
        what: str = "call",
    ):
        """Exponential backoff with jitter for transient errors; works for sync or async callables."""
        initial_wait = self.initial_wait if initial_wait is None else initial_wait
        max_wait = self.max_wait if max_wait is None else max_wait
        max_attempts = self.max_attempts if max_attempts is None else max_attempts
        attempt = 0
        while True:
            try:
                r = coro_factory()
                return await r if inspect.isawaitable(r) else r
            except NON_RETRYABLE:
                raise
            except (
                Exception
            ) as error:  # RpcPolicyError, connection drops, timeouts, node errors
                attempt += 1
                if attempt >= max_attempts:
                    raise RetryExhausted(
                        f"{what} failed after {attempt} attempts: {error!r}"
                    ) from error
                wait = min(
                    initial_wait * 2 ** (attempt - 1), max_wait
                ) * np.random.uniform(0.8, 1.2)
                kind = (
                    "RPC POLICY"
                    if isinstance(error, RpcPolicyError)
                    else type(error).__name__
                )
                self._log(
                    f"[{kind}] {what} attempt={attempt}: {error} -> sleeping {wait:.1f}s"
                )
                await asyncio.sleep(wait)

    # ------------------------------------------------------------------ block list
    def block_collection(
        self, block_amount: int, current_block: int, duration_month: int
    ) -> List[int]:
        # NOTE: 12s/block. Returns block_amount + 1 blocks, oldest first.
        step = int(duration_month * (30 * 24 * 3600) / 12 / block_amount)
        return [current_block - i * step for i in range(block_amount, -1, -1)]

    def cache_block_info(
        self,
        sub,
        block_amount: int,
        current_block: int,
        duration_month: int,
        save_path: str,
    ):
        """`sub` is a synchronous bittensor.Subtensor (as in your run script)."""
        bis = []
        for b in tqdm(
            self.block_collection(block_amount, current_block, duration_month)
        ):
            bi = sub.block_info(b).__dict__
            bi["timestamp"] = bi["timestamp"].isoformat()
            bis.append(bi)
        with open(save_path, "w") as f:
            json.dump(bis, f)
        return bis

    async def cache_block_info_with_retry(
        self,
        sub,
        block_amount: int,
        current_block: int,
        duration_month: int,
        save_path: str,
        max_wait: int = 1800,
        initial_wait: int = 60,
    ):
        bis = []
        for b in tqdm(
            self.block_collection(block_amount, current_block, duration_month)
        ):
            bi = await self.historical_retry(
                lambda b=b: sub.block_info(b),
                initial_wait=initial_wait,
                max_wait=max_wait,
                what=f"block_info {b}",
            )
            bi = bi.__dict__
            bi["timestamp"] = bi["timestamp"].isoformat()
            bis.append(bi)
        with open(save_path, "w") as f:
            json.dump(bis, f)
        return bis

    @staticmethod
    def load_cache_block_info(bi_path: str) -> List[BlockInfo]:
        with open(bi_path) as f:
            data = json.load(f)
        return [
            BlockInfo(
                number=e["number"],
                hash=e["hash"],
                timestamp=datetime.fromisoformat(e["timestamp"]),
            )
            for e in data
        ]

    # ------------------------------------------------------------------ subnet selection
    async def get_subnets_by_async(
        self, metric: str, block_hash: str, cutoff: int = 50
    ):
        sf = StorageFunctions[metric].value
        async with self._open_client() as client:
            io = ChainIO(client._substrate, self)
            items = await io.query_map(sf, [], block_hash)
        scores = {}
        for k, v in items:
            v = plain(v)
            vals = np.array(
                [as_number(x) for x in (v if isinstance(v, list) else [v])], dtype=float
            )
            scores[as_int(k)] = float(np.nansum(vals))
        ranked = sorted(scores, key=lambda n: scores[n], reverse=True)
        return ranked[:cutoff], scores

    def get_subnets_by(
        self,
        metric: Literal["dividends", "emission", "active", "incentive", "subnet_tao"],
        block_hash: str,
        cutoff: int = 50,
    ):
        """Top `cutoff` netuids by summed `metric`. Returns (netuids, {netuid: score}).

        Emission/Dividends are in each subnet's own alpha, so they are not comparable across
        subnets; `subnet_tao` (TAO in the subnet pool) ranks on a common unit."""
        return asyncio.run(self.get_subnets_by_async(metric, block_hash, cutoff))

    # ------------------------------------------------------------------ one block
    def _specs(self, kind: str) -> List[StorageSpec]:
        return [s for s in self.specs if s.kind == kind]

    async def _safe(self, spec: StorageSpec, coro, errors: List[str]):
        try:
            return await coro
        except Exception as e:
            if not spec.optional:
                raise
            errors.append(f"{spec.name}: {e!r}")
            self._log(f"  optional {spec.name} skipped: {e!r}")
            return None

    @staticmethod
    def _shape_value(spec: StorageSpec, v: Any, n: int) -> Any:
        if spec.shape == "uid" or (spec.shape == "auto" and isinstance(plain(v), list)):
            return to_vector(v, n)
        if spec.shape == "raw":
            return plain(v)
        return decode(v, spec)

    async def collect_block(
        self, io: ChainIO, bi: BlockInfo, netuids: List[int]
    ) -> BlockSnapshot:
        bh = bi.hash
        snap = BlockSnapshot(block_info=bi, netuids=list(netuids))
        errs: List[str] = []

        # 1) maps keyed by netuid: one batched call per storage item across all subnets
        subnet_specs = self._specs("subnet")
        batches = await asyncio.gather(
            *(
                self._safe(
                    sp, io.query_batch(sp.item, [[n] for n in netuids], bh), errs
                )
                for sp in subnet_specs
            )
        )
        raw = {
            sp.name: (sp, b) for sp, b in zip(subnet_specs, batches) if b is not None
        }
        for i, net in enumerate(netuids):
            vals = {name: (sp, b[i]) for name, (sp, b) in raw.items()}
            n = as_number(vals["SubnetworkN"][1]) if "SubnetworkN" in vals else np.nan
            if not np.isfinite(n) or n <= 0:  # fall back to the longest per-uid vector
                n = max(
                    (
                        len(plain(v))
                        for _, v in vals.values()
                        if isinstance(plain(v), list)
                    ),
                    default=0,
                )
            n = int(n)
            mc = (
                as_number(vals["MechanismCountCurrent"][1])
                if "MechanismCountCurrent" in vals
                else np.nan
            )
            sn = SubnetSnapshot(
                netuid=net,
                n_uids=n,
                mechanism_count=int(mc) if np.isfinite(mc) and mc >= 1 else 1,
                errors=list(errs),
            )
            for name, (sp, v) in vals.items():
                sn.fields[name] = self._shape_value(sp, v, n)
            snap.subnets[net] = sn

        # 2) maps keyed by NetUidStorageIndex: one batched call per item over (netuid, mech)
        pairs = [
            (net, m)
            for net in netuids
            for m in range(snap.subnets[net].mechanism_count)
        ]
        for sp in self._specs("mechanism"):
            res = await self._safe(
                sp,
                io.query_batch(sp.item, [[mech_index(n, m)] for n, m in pairs], bh),
                errs,
            )
            if res is None:
                continue
            for (net, m), v in zip(pairs, res):
                sn = snap.subnets[net]
                sn.fields[sp.name if m == 0 else f"{sp.name}@mech{m}"] = (
                    self._shape_value(sp, v, sn.n_uids)
                )

        # 3) double maps keyed by uid: one query_map per subnet (and per mechanism for matrices)
        tasks = []
        for net in netuids:
            sn = snap.subnets[net]
            for sp in self._specs("uid_matrix"):
                for m in range(sn.mechanism_count):
                    tasks.append(self._fill_matrix(io, sp, sn, m, bh))
            for sp in self._specs("uid_vector"):
                tasks.append(self._fill_uid_vector(io, sp, sn, bh))
        await asyncio.gather(*tasks)

        # 4) items keyed by hotkey (need Keys): Axons, Prometheus, certificates, commitments,
        #    stake, parent keys
        tasks = []
        for net in netuids:
            sn = snap.subnets[net]
            for sp in self._specs("hotkey_map"):
                tasks.append(self._fill_hotkey_map(io, sp, sn, bh))
            for sp in self._specs("hotkey_netuid"):
                tasks.append(self._fill_hotkey_netuid(io, sp, sn, bh))
        await asyncio.gather(*tasks)

        # 5) coldkeys: only hotkeys not seen before are queried
        for sp in self._specs("owner"):
            await self._fill_owners(io, sp, snap, bh)

        # 6) network-wide StorageValues
        for sp in self._specs("global"):
            v = await self._safe(sp, io.query(sp.item, [], bh), errs)
            if v is not None:
                snap.globals[sp.name] = decode(v, sp)
        if errs:
            snap.globals["errors"] = errs
        return snap

    async def _fill_matrix(
        self, io: ChainIO, sp: StorageSpec, sn: SubnetSnapshot, mech: int, bh: str
    ):
        items = await self._safe(
            sp, io.query_map(sp.item, [mech_index(sn.netuid, mech)], bh), sn.errors
        )
        if items is None:
            return
        n = sn.n_uids
        M = np.zeros((n, n), dtype=np.uint16)
        dropped = 0
        for k, row in items:
            src = as_int(k)
            for pair in plain(row) or []:
                dest, w = int(pair[0]), int(pair[1])
                if src < n and dest < n:
                    M[src, dest] = w
                else:
                    dropped += 1
        if dropped:
            sn.errors.append(
                f"{sp.name}[mech {mech}]: {dropped} entries outside n_uids={n}"
            )
        (
            sn.weights
            if sp.name == "Weights"
            else (
                sn.bonds
                if sp.name == "Bonds"
                else sn.fields.setdefault(f"{sp.name}_matrix", {})
            )
        )[mech] = M

    async def _fill_uid_vector(
        self, io: ChainIO, sp: StorageSpec, sn: SubnetSnapshot, bh: str
    ):
        items = await self._safe(sp, io.query_map(sp.item, [sn.netuid], bh), sn.errors)
        if items is None:
            return
        n = sn.n_uids
        if sp.name == "Keys":
            hk: List[Optional[str]] = [None] * n
            for k, v in items:
                u = as_int(k)
                if u < n:
                    hk[u] = as_account(v)
            sn.hotkeys = hk
            return
        vec = np.full(n, np.nan)
        for k, v in items:
            u = as_int(k)
            if u < n:
                vec[u] = as_number(decode(v, sp))
        sn.fields[sp.name] = vec

    async def _fill_hotkey_map(
        self, io: ChainIO, sp: StorageSpec, sn: SubnetSnapshot, bh: str
    ):
        """(netuid, hotkey) -> value, aligned to uid through Keys."""
        items = await self._safe(sp, io.query_map(sp.item, [sn.netuid], bh), sn.errors)
        if items is None:
            return
        uid_of = {h: u for u, h in enumerate(sn.hotkeys) if h}
        rec: List[Any] = [None] * sn.n_uids
        orphans = 0
        for k, v in items:
            u = uid_of.get(as_account(k))
            if u is None:
                orphans += 1  # entry for a hotkey no longer registered on this subnet
                continue
            rec[u] = decode(v, sp)
        sn.records[sp.name] = rec
        if orphans:
            sn.fields[f"{sp.name}_orphans"] = float(orphans)

    async def _fill_hotkey_netuid(
        self, io: ChainIO, sp: StorageSpec, sn: SubnetSnapshot, bh: str
    ):
        """(hotkey, netuid) -> value for every registered uid."""
        idx = [u for u, h in enumerate(sn.hotkeys) if h]
        if not idx:
            sn.errors.append(f"{sp.name}: no hotkeys (collect Keys first)")
            return
        res = await self._safe(
            sp,
            io.query_batch(sp.item, [[sn.hotkeys[u], sn.netuid] for u in idx], bh),
            sn.errors,
        )
        if res is None:
            return
        vals = [decode(v, sp) for v in res]
        if sp.name == "TotalHotkeyAlpha":
            stake = np.zeros(sn.n_uids)
            for u, v in zip(idx, vals):
                stake[u] = np.nan_to_num(as_number(v))
            sn.stake = stake
            sn.fields["TotalHotkeyAlpha"] = stake
        else:
            rec: List[Any] = [None] * sn.n_uids
            for u, v in zip(idx, vals):
                rec[u] = v
            sn.records[sp.name] = rec

    async def _fill_owners(
        self, io: ChainIO, sp: StorageSpec, snap: BlockSnapshot, bh: str
    ):
        """Owner(hotkey) -> coldkey. A hotkey's owner only changes through a swap, which
        registers a new hotkey, so each hotkey is queried once and cached."""
        new = sorted(
            {
                h
                for sn in snap.subnets.values()
                for h in sn.hotkeys
                if h and h not in self.owner_cache
            }
        )
        if new:
            res = await self._safe(
                sp,
                io.query_batch(sp.item, [[h] for h in new], bh),
                snap.globals.setdefault("errors", []),
            )
            if res is not None:
                for h, c in zip(new, res):
                    self.owner_cache[h] = as_account(c)
        if not snap.globals.get("errors"):
            snap.globals.pop("errors", None)
        for sn in snap.subnets.values():
            sn.coldkeys = [self.owner_cache.get(h) if h else None for h in sn.hotkeys]

    # ------------------------------------------------------------------ many blocks
    @staticmethod
    def _block_path(outdir: str, number: int) -> str:
        return os.path.join(outdir, f"block_{int(number)}.json.gz")

    @staticmethod
    def collected_blocks(outdir: str) -> List[int]:
        return _collected_blocks(outdir)

    def save_snapshot(self, snap: BlockSnapshot, outdir: str) -> str:
        os.makedirs(outdir, exist_ok=True)
        path = self._block_path(outdir, snap.block_info.number)
        tmp = path + ".tmp"
        with gzip.open(tmp, "wt", compresslevel=5) as f:
            json.dump(snapshot_to_dict(snap), f)
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written block
        return path

    def _load_owner_cache(self, outdir: str):
        p = os.path.join(outdir, "owners.json")
        if os.path.exists(p):
            with open(p) as f:
                self.owner_cache.update(json.load(f))

    def _save_owner_cache(self, outdir: str):
        p = os.path.join(outdir, "owners.json")
        with open(p + ".tmp", "w") as f:
            json.dump(self.owner_cache, f)
        os.replace(p + ".tmp", p)

    async def collect_blocksnapshots_with_retry(
        self,
        block_info: List[BlockInfo],
        netuids: List[int],
        checkpoint: bool = True,
        outpath: str = "./exports/blocks/",
        return_snapshots: bool = False,
    ) -> List[BlockSnapshot]:
        """Collect every block in `block_info` for `netuids`.

        With checkpoint=True each block is written to `outpath/block_<number>.json.gz` as
        soon as it is complete, and blocks already there are skipped, so re-running the
        same call resumes. Set return_snapshots=True to also get them back in memory
        (dense matrices for 60 subnets x 500 blocks need several GB).
        """
        done = set(self.collected_blocks(outpath)) if checkpoint else set()
        todo = [bi for bi in block_info if int(bi.number) not in done]
        if checkpoint:
            self._load_owner_cache(outpath)
        kept: List[BlockSnapshot] = []
        bar = tqdm(
            total=len(block_info), initial=len(block_info) - len(todo), desc="blocks"
        )
        reconnects = 0
        while todo:
            try:
                async with self._open_client() as client:
                    io = ChainIO(client._substrate, self)
                    while todo:
                        snap = await self.collect_block(io, todo[0], netuids)
                        if checkpoint:
                            self.save_snapshot(snap, outpath)
                            self._save_owner_cache(outpath)
                        if return_snapshots or not checkpoint:
                            kept.append(snap)
                        todo.pop(0)
                        bar.update(1)
                        reconnects = 0
            except NON_RETRYABLE:
                bar.close()
                raise
            except (
                Exception
            ) as e:  # dropped connection or retries exhausted: reopen and resume
                reconnects += 1
                if reconnects > self.max_reconnects:
                    bar.close()
                    raise RetryExhausted(
                        f"gave up after {reconnects} reconnects at block "
                        f"{todo[0].number}: {e!r}"
                    ) from e
                wait = min(self.initial_wait * 2**reconnects, self.max_wait)
                self._log(
                    f"[RECONNECT {reconnects}] block {todo[0].number}: {e!r} -> sleeping {wait:.0f}s"
                )
                await asyncio.sleep(wait)
        bar.close()
        if checkpoint and return_snapshots:
            return self.load_blocksnapshots(
                outpath, blocks=[int(b.number) for b in block_info]
            )
        return kept

    # ------------------------------------------------------------------ identity (Owner, Axons)
    def export_identity(
        self, inpath: str, outpath: str, n_points: int = 5
    ) -> List[Dict]:
        """Coldkeys and axon IPs at `n_points` evenly spaced collected blocks, in the
        format bt_analysis reads with --identity. Uses the collected snapshots only (Owner
        and Axons are collected every block now), so it makes no chain calls."""
        nums = self.collected_blocks(inpath)
        if not nums:
            return []
        picks = sorted(
            {
                nums[int(i)]
                for i in np.linspace(0, len(nums) - 1, min(n_points, len(nums))).round()
            }
        )
        recs = []
        for snap in self.load_blocksnapshots(inpath, blocks=picks):
            for net, sn in snap.subnets.items():
                ips = sn.axon_ips()
                recs.append(
                    {
                        "block_number": int(snap.block_info.number),
                        "netuid": int(net),
                        "axons": {
                            h: {"ip": ip} for h, ip in zip(sn.hotkeys, ips) if h and ip
                        },
                        "owners": {
                            h: c for h, c in zip(sn.hotkeys, sn.coldkeys) if h and c
                        },
                    }
                )
        with open(outpath, "w") as f:
            json.dump(recs, f)
        return recs

    # ------------------------------------------------------------------ loading / export
    def load_blocksnapshots(
        self,
        inpath: str,
        blocks: Optional[Iterable[int]] = None,
        netuids: Optional[List[int]] = None,
    ) -> List[BlockSnapshot]:
        """Load per-block files from a directory (optionally only some blocks / subnets)."""
        return _load_blocksnapshots(
            inpath, blocks=list(blocks) if blocks is not None else None, netuids=netuids
        )

    def export_for_analysis(self, inpath: str, outpath: str, mech: int = 0) -> int:
        """Write the single-file layout bt_analysis reads (fields aligned with `netuid`).
        Matrices are written sparse ([[src, [[dest, w], ...]], ...]). Streams block by block.
        """
        nums = self.collected_blocks(inpath)
        with open(outpath, "w") as f:
            f.write("[")
            for i, n in enumerate(tqdm(nums, desc="export")):
                with gzip.open(self._block_path(inpath, n), "rt") as g:
                    d = json.load(g)
                f.write(("," if i else "") + json.dumps(_legacy_entry(d, mech)))
            f.write("]")
        return len(nums)


def _rec_field(s: Optional[Dict], name: str, key: str) -> List:
    if not s:
        return []
    rec = s.get("records", {}).get(name) or [None] * s["n_uids"]
    return [(r.get(key) if isinstance(r, dict) else None) for r in rec]


def _legacy_entry(d: Dict, mech: int) -> Dict:
    """One block in the single-file layout: every field is a list aligned with `netuid`."""
    nets = [str(n) for n in d["netuids"]]
    subs = [d["subnets"].get(n) for n in nets]
    e = {"block_info": d["block_info"], "netuid": [int(n) for n in nets]}
    names = set()
    for s in subs:
        if s:
            names |= set(s["fields"])
    for name in sorted(names):
        if "@mech" in name:
            continue
        e[name] = [(s["fields"].get(name) if s else None) for s in subs]
    if mech:
        for name in ("Incentive", "LastUpdate"):
            e[name] = [
                (
                    s["fields"].get(f"{name}@mech{mech}", s["fields"].get(name))
                    if s
                    else None
                )
                for s in subs
            ]
    e["Weights"] = [(s["weights"].get(str(mech), []) if s else []) for s in subs]
    e["Bonds"] = [(s["bonds"].get(str(mech), []) if s else []) for s in subs]
    e["Keys"] = [(s["hotkeys"] if s else []) for s in subs]
    e["Coldkeys"] = [(s.get("coldkeys", []) if s else []) for s in subs]
    e["AxonIP"] = [_rec_field(s, "Axons", "ip") for s in subs]
    e["AxonPort"] = [_rec_field(s, "Axons", "port") for s in subs]
    e["AxonBlock"] = [_rec_field(s, "Axons", "block") for s in subs]
    e["AxonVersion"] = [_rec_field(s, "Axons", "version") for s in subs]
    e["PrometheusBlock"] = [_rec_field(s, "Prometheus", "block") for s in subs]
    e["CommitmentBlock"] = [_rec_field(s, "CommitmentOf", "block") for s in subs]
    certs = [
        (s.get("records", {}).get("NeuronCertificates") or []) if s else []
        for s in subs
    ]
    e["NeuronCertificates"] = [
        [[h, c] for h, c in zip(s["hotkeys"], cs) if h and c] if s else []
        for s, cs in zip(subs, certs)
    ]
    e["ParentKeys"] = [
        (s.get("records", {}).get("ParentKeys") if s else None) for s in subs
    ]
    e["TotalHotkeyAlpha"] = [(s["stake"] if s else None) for s in subs]
    for k, v in d.get("globals", {}).items():
        if k != "errors":
            e[k] = v
    return e
