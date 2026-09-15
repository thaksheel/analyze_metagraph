import numpy as np
import pandas as pd
import json
import time
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Literal
from datetime import datetime
import asyncio
import bittensor
from bittensor.metagraph import Metagraph
from bittensor import Subtensor

from .utils import BlockSnapshot, BlockInfo, StorageFunctions


class MetagraphManager:
    def __init__(self, display: bool = False):
        self.display = display

    def block_collection(
        self,
        block_amount: int,
        current_block: int,
        duration_month: int,
    ):
        # NOTE: 12s/block
        blocks_per_amount = int(duration_month * (30 * 24 * 3600) / 12 / block_amount)
        blocks = [
            current_block - i * blocks_per_amount for i in range(block_amount, -1, -1)
        ]
        return blocks

    def cache_block_info(
        self,
        sub: Subtensor,
        block_amount: int,
        current_block: int,
        duration_month: int,
        save_path: str,
    ):
        blocks = self.block_collection(block_amount, current_block, duration_month)
        bis = [sub.block_info(b).__dict__ for b in blocks]
        bis = []
        for b in blocks:
            bi = sub.block_info(b).__dict__
            bi["timestamp"] = bi["timestamp"].isoformat()
            bis.append(bi)
        with open(save_path, "w") as f:
            json.dump(bis, f)
        return bis 

    def cache_blocksnapshots(self, snapshots: List[BlockSnapshot], outpath:str):
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if hasattr(obj, "__dict__"):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            return obj
        data = [convert(s) for s in snapshots]
        with open(outpath, "w") as f:
            json.dump(data, f)
        return self 

    def load_blocksnapshots(self, inpath: str): 
        def to_array_list(x):
            return [np.array(v) for v in x]

        with open(inpath, "r") as f:
            raw = json.load(f)
        snapshots = []
        for entry in raw:
            bi_dict = entry["block_info"]
            block_info = BlockInfo(**bi_dict)            
            snapshot = BlockSnapshot(
                block_info=block_info,
                netuid=entry["netuid"],
                Active=to_array_list(entry["Active"]),
                ActivityCutoff=to_array_list(entry["ActivityCutoff"]),
                Bonds=to_array_list(entry["Bonds"]),
                Consensus=to_array_list(entry["Consensus"]),
                Dividends=to_array_list(entry["Dividends"]),
                Emission=to_array_list(entry["Emission"]),
                Incentive=to_array_list(entry["Incentive"]),
                Kappa=to_array_list(entry["Kappa"]),
                LastUpdate=to_array_list(entry["LastUpdate"]),
                MechanismCountCurrent=to_array_list(entry["MechanismCountCurrent"]),
                NeuronCertificates=to_array_list(entry["NeuronCertificates"]),
                Rho=to_array_list(entry["Rho"]),
                SubnetMechanism=to_array_list(entry["SubnetMechanism"]),
                Tempo=to_array_list(entry["Tempo"]),
                ValidatorPermit=to_array_list(entry["ValidatorPermit"]),
                ValidatorTrust=to_array_list(entry["ValidatorTrust"]),
                Weights=to_array_list(entry["Weights"]),
            )
            snapshots.append(snapshot)
        return snapshots 

    def load_cache_block_info(self, bi_path: str) -> List[BlockInfo]:
        """Collects only the hash, number, and timestamps as `BlockInfo` fields while ignoring the rest."""
        with open(bi_path, "r") as f:
            data = json.load(f)
        blocks = []
        for entry in data:
            blocks.append(
                BlockInfo(
                    number=entry["number"],
                    hash=entry["hash"],
                    timestamp=datetime.fromisoformat(entry["timestamp"]),
                )
            )
        return blocks

    def get_subnets_by(
        self,
        metric: Literal["dividends", "emission", "active"],
        block_hash: str,
        cutoff: int = 50,
    ) -> List[int]:
        sf = StorageFunctions[metric].value
        netuids, d = asyncio.run(self.fetch_all_netuids(sf, block_hash))
        data = np.array([d.sum() for d in d])
        scored = []
        for netuid in netuids:
            values = np.array(data[netuid])
            if values.size == 0:
                score = -np.inf
            else:
                score = float(values.mean())
            scored.append((netuid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        sorted_netuids = [int(n) for n, _ in scored]
        return sorted_netuids[:cutoff], d

    async def collect_blocksnapshots(
        self, block_info: List[BlockInfo], netuids: List[int]
    ):
        client = bittensor.Client(network="finney")
        await client.connect()
        subs = client._substrate
        blocksnapshots: List[BlockSnapshot] = []
        netuids_lst = [[i] for i in netuids]
        for bi in tqdm(block_info):
            snap = BlockSnapshot(block_info=bi, netuid=netuids)
            for sf in StorageFunctions:
                data = await self.fetch_by_netuids(
                    storage_func=sf.value,
                    block_hash=bi.hash,
                    netuids_params=netuids_lst,
                    substrate=subs,
                )
                if data is None:
                    raise ValueError(
                        "fetch data from query batch is not working. See `fetch_by_netuids`"
                    )
                data = [np.array(d) for d in data]
                setattr(snap, sf.value.lower(), data)
            blocksnapshots.append(snap)
        return blocksnapshots

    async def fetch_by_netuids(
        self,
        storage_func: str,
        block_hash: str,
        netuids_params: List[List[str]],
        substrate: bittensor.Substrate,
    ):
        try:
            data = await substrate.query_batch(
                module="SubtensorModule",
                storage_function=storage_func,
                block_hash=block_hash,
                param_sets=netuids_params,
            )
            return data
        except:
            print(f"--->Error in collecting {storage_func} data. Review this!!!")
            return None

    async def fetch_all_netuids(self, storage_func: str, block_hash: str):
        client = bittensor.Client(network="finney")
        await client.connect()
        subs = client._substrate
        try:
            value = await subs.query_map(
                module="SubtensorModule",
                storage_function=storage_func,
                block_hash=block_hash,
            )
            netuids = np.array([v[0] for v in value])
            data = [np.array(val[1]) for val in value]
        except:
            print(f"---> {storage_func} exception")
            return None
        return netuids, data

