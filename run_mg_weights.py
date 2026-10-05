import asyncio
import time
from bittensor import Subtensor

from src import StorageFunctions
from src.metagraph_fetch import MetagraphManager

mm = MetagraphManager()
sub = Subtensor(
    network="finney",
    archive_endpoints=["wss://archive.chain.opentensor.ai:443"],
)
bis = mm.load_cache_block_info(bi_path="./exports/block_infos.json")
current_bh = "0x5ad174d4fa5fef3147d5fb61cca5f3da6eb1dede2f8c8d3288fb4b3160f8bf06"
weights = asyncio.run(
    mm.fetch_by_netuids(
        storage_func="StakeWeight",
        block_hash=current_bh,
        netuids_params=[[20]],
        substrate=sub,
    )
)

weights_2 = asyncio.run(
    mm.fetch_all_netuids(
        storage_func="Weights",
        block_hash=current_bh,
    )
)

print(len(weights))
print("END")
