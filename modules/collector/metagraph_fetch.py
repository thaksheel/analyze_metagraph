import asyncio
import inspect
import json
import time
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

from tqdm import tqdm
import numpy as np

try:
    import bittensor
except ImportError:  # allows offline tests
    bittensor = None
try:
    from bittensor.result import RpcPolicyError
except Exception:

    class RpcPolicyError(Exception):
        pass


from .utils import (
    DEFAULT_GROUPS,
    ITEMS,
    MECHANISM_STRIDE,
    BlockInfo,
    BlockSnapshot,
    Item,
    StorageFunctions,
    as_account,
    as_number,
    collected_blocks,
    decode,
    load_blocksnapshots,
    merge_blocks,
    plain,
    save_block,
)
from .utils import load_cache_block_info as _load_cache_block_info

REQUIRED = {"Emission", "Incentive"}
NON_RETRYABLE = (TypeError, KeyError, ValueError, AttributeError, NotImplementedError)


class RetryExhausted(RuntimeError):
    pass


class MetagraphManager:
    def __init__(
        self,
        display: bool = False,
        network: str = "finney",
        include: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        every: Optional[Dict[str, int]] = None,
        all_mechanisms: bool = False,
        batch_size: int = 2000,
        initial_wait: float = 60,
        max_wait: float = 1800,
        max_attempts: int = 10,
        max_reconnects: int = 20,
        client_factory: Optional[Callable] = None,
    ):
        """
        include         groups to collect (default: core, economics, weights, keys).
                        All groups: core, economics, weights, bonds, keys, stake, owner,
                        axons, certificates.
        exclude_items   storage names to drop even if their group is included
        every           {storage name: N} to fetch an item every N-th block only
        all_mechanisms  also collect Weights/Bonds/Incentive/LastUpdate for mechanisms > 0
        batch_size      max keys per query_batch call (bigger items are split)
        client_factory  async callable returning a connected client with `._substrate`
                        (tests, or a different client class)
        """
        self.display = display
        self.network = network
        groups = set(include or DEFAULT_GROUPS)
        exclude = set(exclude_items or [])
        cadence = every or {}
        self.items: List[Item] = []
        for it in ITEMS:
            if it.group in groups and it.name not in exclude:
                n = cadence.get(it.name, cadence.get(it.field, it.every))
                self.items.append(
                    Item(
                        it.name, it.kind, it.group, it.decode, int(n), it.module, it.out
                    )
                )
        needs_keys = {"hotkey", "hotkey_net", "account"}
        if any(i.kind in needs_keys for i in self.items) and not any(
            i.name == "Keys" for i in self.items
        ):
            raise ValueError(
                "stake / owner / axons / certificates need the 'keys' group"
            )
        self.all_mechanisms = all_mechanisms
        self.batch_size = batch_size
        self.initial_wait = initial_wait
        self.max_wait = max_wait
        self.max_attempts = max_attempts
        self.max_reconnects = max_reconnects
        self.client_factory = client_factory
        self.owner_cache: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------ connection
    async def _connect(self):
        if self.client_factory is not None:
            client = self.client_factory()
            client = await client if inspect.isawaitable(client) else client
        else:
            client = bittensor.Client(network=self.network)
            await client.connect()
        return client, client._substrate

    @staticmethod
    async def _close(client):
        for name in ("close", "disconnect"):
            fn = getattr(client, name, None)
            if callable(fn):
                try:
                    r = fn()
                    if inspect.isawaitable(r):
                        await r
                except Exception:
                    pass
                return

    def _log(self, msg: str):
        if self.display:
            print(msg, flush=True)

    async def historical_retry(
        self,
        coro_factory,
        initial_wait: Optional[float] = None,
        max_wait: Optional[float] = None,
        what: str = "call",
    ):
        """Backoff with jitter for rate limits / connection errors, bounded by max_attempts."""
        initial_wait = self.initial_wait if initial_wait is None else initial_wait
        max_wait = self.max_wait if max_wait is None else max_wait
        attempt = 0
        while True:
            try:
                r = coro_factory()
                return await r if inspect.isawaitable(r) else r
            except NON_RETRYABLE:
                raise
            except Exception as error:
                attempt += 1
                if attempt >= self.max_attempts:
                    raise RetryExhausted(
                        f"{what}: {attempt} attempts, last error {error!r}"
                    ) from error
                wait = min(
                    initial_wait * 2 ** (attempt - 1), max_wait
                ) * np.random.uniform(0.8, 1.2)
                tag = (
                    "RPC POLICY RATE LIMIT"
                    if isinstance(error, RpcPolicyError)
                    else type(error).__name__
                )
                self._log(
                    f"\n [{tag}] {what} attempt={attempt}: {error}\n sleeping {wait:.1f}s"
                )
                await asyncio.sleep(wait)

    async def _batch(
        self, subs, item: Item, param_sets: List[List], block_hash: str
    ) -> List[Any]:
        out: List[Any] = []
        for s in range(0, len(param_sets), self.batch_size):
            chunk = param_sets[s : s + self.batch_size]
            res = await self.historical_retry(
                lambda c=chunk: subs.query_batch(
                    module=item.module,
                    storage_function=item.name,
                    block_hash=block_hash,
                    param_sets=c,
                ),
                what=f"{item.name} x{len(chunk)}",
            )
            res = list(res)
            if len(res) != len(chunk):
                raise ValueError(
                    f"{item.name}: query_batch returned {len(res)} values for {len(chunk)} keys"
                )
            out.extend(res)
        return out

    # ------------------------------------------------------------------ block info / subnets
    def block_collection(
        self, block_amount: int, current_block: int, duration_month: int
    ) -> List[int]:
        # NOTE: 12s/block; returns block_amount + 1 blocks, oldest first
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
            # sub.block_info is synchronous; historical_retry handles both kinds
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

    def load_cache_block_info(self, bi_path: str) -> List[BlockInfo]:
        return _load_cache_block_info(bi_path)

    async def fetch_all_netuids(self, storage_func: str, block_hash: str):
        client, subs = await self._connect()
        try:
            value = await self.historical_retry(
                lambda: subs.query_map(
                    module="SubtensorModule",
                    storage_function=storage_func,
                    block_hash=block_hash,
                ),
                what=f"query_map {storage_func}",
            )
            value = (
                [v async for v in value] if hasattr(value, "__aiter__") else list(value)
            )
            netuids = [int(as_number(v[0])) for v in value]
            data = [plain(v[1]) for v in value]
        finally:
            await self._close(client)
        return netuids, data

    def get_subnets_by(
        self,
        metric: Literal["dividends", "emission", "active", "incentive", "subnet_tao"],
        block_hash: str,
        cutoff: int = 50,
    ):
        """Top `cutoff` netuids by the summed metric -> (netuids, {netuid: score}).
        Emission/Dividends are in each subnet's own alpha; `subnet_tao` ranks on TAO."""
        netuids, data = asyncio.run(
            self.fetch_all_netuids(StorageFunctions[metric].value, block_hash)
        )
        scores = {}
        for n, d in zip(netuids, data):  # keyed by netuid, not by list position
            vals = [as_number(x) for x in (d if isinstance(d, list) else [d])]
            scores[n] = (
                float(np.nansum([v for v in vals if v is not None]))
                if vals
                else -np.inf
            )
        ranked = sorted(scores, key=lambda n: scores[n], reverse=True)
        return ranked[:cutoff], scores

    # ------------------------------------------------------------------ one block
    def _due(self, item: Item, block_index: int) -> bool:
        return item.every <= 1 or block_index % item.every == 0

    async def collect_block(
        self,
        subs,
        bi: BlockInfo,
        netuids: List[int],
        block_index: int = 0,
        force_all: bool = False,
    ) -> BlockSnapshot:
        snap = BlockSnapshot(block_info=bi, netuid=list(netuids))
        F = snap.fields
        bh = bi.hash
        items = [i for i in self.items if force_all or self._due(i, block_index)]
        by_kind = {
            k: [i for i in items if i.kind == k]
            for k in (
                "subnet",
                "mechanism",
                "uid",
                "validator",
                "hotkey",
                "hotkey_net",
                "account",
                "global",
            )
        }

        async def run(item: Item, coro):
            t0 = time.perf_counter()
            try:
                return await coro
            except Exception as e:
                if item.name in REQUIRED:
                    raise
                snap.errors.append(f"{item.name}: {e!r}")
                self._log(f"  [skip] {item.name}: {e!r}")
                return None
            finally:
                snap.timings[item.name] = round(time.perf_counter() - t0, 3)

        # 1) subnet maps: one call per item, 60 keys
        for it in by_kind["subnet"]:
            res = await run(it, self._batch(subs, it, [[n] for n in netuids], bh))
            if res is not None:
                F[it.field] = [decode(v, it.decode) for v in res]
        n_uids = [self._n(F, i) for i in range(len(netuids))]
        mechs = [
            (
                int(F.get("MechanismCountCurrent", [1] * len(netuids))[i] or 1)
                if self.all_mechanisms
                else 1
            )
            for i in range(len(netuids))
        ]
        pairs = [(i, m) for i in range(len(netuids)) for m in range(mechs[i])]

        # 2) mechanism-indexed vectors
        for it in by_kind["mechanism"]:
            res = await run(
                it,
                self._batch(
                    subs,
                    it,
                    [[m * MECHANISM_STRIDE + netuids[i]] for i, m in pairs],
                    bh,
                ),
            )
            if res is None:
                continue
            for (i, m), v in zip(pairs, res):
                F.setdefault(
                    it.field if m == 0 else f"{it.field}@mech{m}", [None] * len(netuids)
                )[i] = decode(v, it.decode)
        n_uids = [self._n(F, i) for i in range(len(netuids))]

        # 3) per-uid double maps: one call, one key per uid
        for it in by_kind["uid"]:
            keys = [(i, u) for i in range(len(netuids)) for u in range(n_uids[i])]
            res = await run(
                it, self._batch(subs, it, [[netuids[i], u] for i, u in keys], bh)
            )
            if res is not None:
                F[it.field] = self._per_uid(keys, res, it, n_uids)

        # 4) Weights / Bonds for validator uids only
        vp = F.get("ValidatorPermit")
        for it in by_kind["validator"]:
            keys = []
            for i, m in pairs:
                perm = (
                    vp[i] if vp and vp[i] else [1] * n_uids[i]
                )  # no permits known: query all uids
                keys += [(i, m, u) for u, p in enumerate(perm[: n_uids[i]]) if p]
            res = await run(
                it,
                self._batch(
                    subs,
                    it,
                    [[m * MECHANISM_STRIDE + netuids[i], u] for i, m, u in keys],
                    bh,
                ),
            )
            if res is None:
                continue
            for (i, m, u), v in zip(keys, res):
                rows = F.setdefault(
                    it.field if m == 0 else f"{it.field}@mech{m}", [[] for _ in netuids]
                )
                p = decode(v, "pairs")
                if p:
                    rows[i].append([u, p])

        # 5) hotkey-keyed items (use this block's Keys)
        hot = F.get("Keys")
        if hot is None and (
            by_kind["hotkey"] or by_kind["hotkey_net"] or by_kind["account"]
        ):
            snap.errors.append("Keys missing: hotkey items skipped")
            hot = [[] for _ in netuids]
        hkeys = (
            [
                (i, u, h)
                for i in range(len(netuids))
                for u, h in enumerate(hot[i] or [])
                if h
            ]
            if hot
            else []
        )
        for it in by_kind["hotkey"]:
            res = await run(
                it, self._batch(subs, it, [[netuids[i], h] for i, _, h in hkeys], bh)
            )
            if res is not None:
                F[it.field] = self._per_uid(
                    [(i, u) for i, u, _ in hkeys], res, it, n_uids
                )
        for it in by_kind["hotkey_net"]:
            res = await run(
                it, self._batch(subs, it, [[h, netuids[i]] for i, _, h in hkeys], bh)
            )
            if res is not None:
                F[it.field] = self._per_uid(
                    [(i, u) for i, u, _ in hkeys], res, it, n_uids
                )

        # 6) Owner: only hotkeys not seen before
        for it in by_kind["account"]:
            new = sorted({h for _, _, h in hkeys if h not in self.owner_cache})
            res = (
                await run(it, self._batch(subs, it, [[h] for h in new], bh))
                if new
                else []
            )
            if res is not None:
                for h, v in zip(new, res):
                    self.owner_cache[h] = decode(v, it.decode)
            F[it.field] = [
                [self.owner_cache.get(h) if h else None for h in (hot[i] or [])]
                for i in range(len(netuids))
            ]

        # 7) network totals
        for it in by_kind["global"]:
            t0 = time.perf_counter()
            try:
                v = await self.historical_retry(
                    lambda: subs.query(it.module, it.name, [], block_hash=bh),
                    what=it.name,
                )
                F[it.field] = decode(v, it.decode)
            except Exception as e:
                snap.errors.append(f"{it.name}: {e!r}")
            snap.timings[it.name] = round(time.perf_counter() - t0, 3)
        return snap

    @staticmethod
    def _n(F: Dict, i: int) -> int:
        n = (F.get("SubnetworkN") or [None] * (i + 1))[i]
        if n:
            return int(n)
        return max(
            (len(F[k][i] or []) for k in ("Emission", "Incentive") if F.get(k)),
            default=0,
        )

    @staticmethod
    def _per_uid(
        keys: List[Tuple[int, int]], res: List[Any], it: Item, n_uids: List[int]
    ) -> List[List]:
        out = [[None] * n for n in n_uids]
        for (i, u), v in zip(keys, res):
            if u < n_uids[i]:
                out[i][u] = decode(v, it.decode)
        return out

    # ------------------------------------------------------------------ many blocks
    async def collect_blocksnapshots_with_retry(
        self,
        block_info: List[BlockInfo],
        netuids: List[int],
        checkpoint: bool = True,
        outpath: str = "./exports/blocks/",
    ) -> List[BlockSnapshot]:
        """Collect all blocks; each is saved to `outpath/block_<n>.json.gz` as soon as it is
        done and skipped on the next run. Returns snapshots only when checkpoint=False.
        """
        done = set(collected_blocks(outpath)) if checkpoint else set()
        if checkpoint:
            self._load_owner_cache(outpath)
        todo = [
            (k, bi) for k, bi in enumerate(block_info) if int(bi.number) not in done
        ]
        kept: List[BlockSnapshot] = []
        bar = tqdm(
            total=len(block_info), initial=len(block_info) - len(todo), desc="blocks"
        )
        reconnects = 0
        while todo:
            client, subs = await self._connect()
            try:
                while todo:
                    k, bi = todo[0]
                    t0 = time.perf_counter()
                    snap = await self.collect_block(subs, bi, netuids, block_index=k)
                    if checkpoint:
                        save_block(snap, outpath)
                        self._save_owner_cache(outpath)
                    else:
                        kept.append(snap)
                    todo.pop(0)
                    reconnects = 0
                    bar.update(1)
                    slow = sorted(snap.timings.items(), key=lambda x: -x[1])[:3]
                    bar.set_postfix_str(
                        f"{time.perf_counter() - t0:.0f}s/block; slowest "
                        + ", ".join(f"{n} {s:.0f}s" for n, s in slow)
                    )
            except NON_RETRYABLE:
                bar.close()
                raise
            except (
                Exception
            ) as e:  # retries exhausted or connection dropped: reconnect, resume
                reconnects += 1
                if reconnects > self.max_reconnects:
                    bar.close()
                    raise RetryExhausted(
                        f"gave up at block {todo[0][1].number}: {e!r}"
                    ) from e
                wait = min(self.initial_wait * 2 ** (reconnects - 1), self.max_wait)
                self._log(
                    f"\n [RECONNECT {reconnects}] block {todo[0][1].number}: {e!r}; sleeping {wait:.0f}s"
                )
                await asyncio.sleep(wait)
            finally:
                await self._close(client)
        bar.close()
        return kept

    def _load_owner_cache(self, d: str):
        import os

        p = os.path.join(d, "owners.json")
        if os.path.exists(p):
            with open(p) as f:
                self.owner_cache.update(json.load(f))

    def _save_owner_cache(self, d: str):
        import os

        if not self.owner_cache:
            return
        p = os.path.join(d, "owners.json")
        with open(p + ".tmp", "w") as f:
            json.dump(self.owner_cache, f)
        os.replace(p + ".tmp", p)

    # ------------------------------------------------------------------ benchmark
    async def benchmark_async(
        self, bi: BlockInfo, netuids: List[int], n_blocks: int = 501
    ) -> List[Dict]:
        """Time every selected item on one block (cadence ignored) and estimate the full run."""
        client, subs = await self._connect()
        try:
            snap = await self.collect_block(subs, bi, netuids, force_all=True)
        finally:
            await self._close(client)
        rows = []
        for it in self.items:
            sec = snap.timings.get(it.name, 0.0)
            fetches = int(np.ceil(n_blocks / max(it.every, 1)))
            rows.append(
                {
                    "item": it.name,
                    "group": it.group,
                    "seconds_one_block": sec,
                    "every": it.every,
                    "estimated_hours": sec * fetches / 3600,
                }
            )
        rows.sort(key=lambda r: -r["estimated_hours"])
        total = sum(r["estimated_hours"] for r in rows)
        print(f"\n{'item':<30}{'group':<14}{'s/block':>9}{'every':>7}{'est. h':>9}")
        for r in rows:
            print(
                f"{r['item']:<30}{r['group']:<14}{r['seconds_one_block']:>9.2f}{r['every']:>7}{r['estimated_hours']:>9.2f}"
            )
        print(
            f"{'total':<30}{'':<14}{'':>9}{'':>7}{total:>9.2f}  (for {n_blocks} blocks)"
        )
        if snap.errors:
            print("errors:", *snap.errors, sep="\n  ")
        return rows

    def benchmark(
        self, bi: BlockInfo, netuids: List[int], n_blocks: int = 501
    ) -> List[Dict]:
        return asyncio.run(self.benchmark_async(bi, netuids, n_blocks))

    # ------------------------------------------------------------------ loading / export
    def load_blocksnapshots(
        self, inpath: str, blocks: Optional[Iterable[int]] = None
    ) -> List[BlockSnapshot]:
        return load_blocksnapshots(inpath, list(blocks) if blocks is not None else None)

    def cache_blocksnapshots(self, inpath: str, outpath: str) -> int:
        """Merge the per-block files into one JSON list (the bt_analysis input)."""
        return merge_blocks(inpath, outpath)
