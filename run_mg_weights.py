import asyncio
import time
from bittensor import Subtensor
from matplotlib import pyplot as plt 

from src.metagraph_fetch import MetagraphManager

mm = MetagraphManager()
sub = Subtensor(
    network="finney",
    archive_endpoints=["wss://archive.chain.opentensor.ai:443"],
)
# bis = mm.load_cache_block_info(bi_path="./exports/block_infos.json")
current_bh = "0x5ad174d4fa5fef3147d5fb61cca5f3da6eb1dede2f8c8d3288fb4b3160f8bf06"
BLOCK_HASH = "0x5ad174d4fa5fef3147d5fb61cca5f3da6eb1dede2f8c8d3288fb4b3160f8bf06"
NETUID, MECHID = 20, 0


async def main():
    async with Subtensor("finney") as client:  # async Client; archive fallback is on
        subs = client._substrate  # or: use block=<number> on client.query_map
        idx = MECHID * 4096 + NETUID  # NetUidStorageIndex
        weights = await subs.query_map(
            "SubtensorModule", "Weights", [idx], block_hash=BLOCK_HASH
        )  # [(uid, [(dest_uid, w_u16), ...]), ...]
        total_stake = await subs.query(
            "SubtensorModule", "TotalStake", [], block_hash=BLOCK_HASH
        )  # StorageValue, rao
        stake_weight = await subs.query(
            "SubtensorModule", "StakeWeight", [idx], block_hash=BLOCK_HASH
        ) 
    print(weights[:3], len(weights))
    print(total_stake)
    return idx, weights, total_stake, stake_weight 

idx, weights, total_stake, stake_weight = asyncio.run(main())

out1 = asyncio.run(
    mm.fetch_by_netuids(
        storage_func="Weights",
        block_hash=current_bh,
        netuids_params=[[20]],
        substrate=sub,
    )
)

out2 = asyncio.run(
    mm.fetch_all_netuids(
        storage_func="TotalStake",
        block_hash=current_bh,
    )
)

print(out1)
print(out2)
print("END")
