import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import asyncio
from enum import Enum
import time
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, List, Optional, Literal, Dict
import bittensor
from bittensor import storage
from bittensor import Subtensor

from src.metagraph_fetch import MetagraphManager
from src.utils import BlockSnapshot, StorageFunctions

# TODO: 1) use archive mode to fetch older blocks, 2) reduce runtime using asycn or query_batch, 3) organize and package all sn info


async def main(sf: str, block_hash: int, block_snapshot: BlockSnapshot):
    client = bittensor.Client(network="finney")
    await client.connect()
    subs = client._substrate
    try:
        value = await subs.query_map(
            module="SubtensorModule",
            storage_function=sf,
            block_hash=block_hash,
            # param_sets=[[0]], # can be 0,1,2
        )
        block_snapshot.netuid = np.array([v[0] for v in value])
        d = []
        for val in value:
            d.append(np.array(val[1]))
        setattr(block_snapshot, sf.lower(), d)
    except:
        print(f"---> {sf} exception")
        return block_snapshot
    return block_snapshot


if __name__ == "__main__":
    durations = [time.time()]
    mm = MetagraphManager()
    stm = storage.SubtensorModule()
    sub = Subtensor(
        network="finney",
        archive_endpoints=["wss://archive.chain.opentensor.ai:443"],
    )
    bis = mm.load_cache_block_info(bi_path="./exports/block_infos.json")

    values = []
    errors = []
    worked = []
    block_data: List[Dict] = []
    for bi in tqdm(bis):
        block_d = BlockSnapshot()
        for sf in StorageFunctions:
            block_d = asyncio.run(
                main(sf=sf.value, block_hash=bi.hash, block_snapshot=block_d)
            )
            if block_d is None:
                errors.append(sf)
            durations.append(time.time())
            print(f"fetch_time for {sf.value}= {durations[-1] - durations[-2]:.2f}s")
        block_data.append(
            {
                "block_num": bi.number,
                "block_timestamp": bi.timestamp,
                "block_snapshot": block_d,
            }
        )
        print(f"\n\nblock_num={bi.number}")

    print(f"Runtime={(time.time() - durations[0]):.2f}s for len_blocks={len(blocks)}")
    print("END")

    # NOTE: explore scripts
    active_sn10 = np.array(values[3][10][1])  # (fields, subnet number, select data)
    active_sn10_uids = np.where(active_sn10 == True)

    div = np.array(values[1][10][1])
    div_active_sn10 = div[active_sn10_uids]
