import asyncio
import time
from bittensor import Subtensor

from src.metagraph_fetch import MetagraphManager

durations = [time.time()]
mm = MetagraphManager()
sub = Subtensor(
    network="finney",
    archive_endpoints=["wss://archive.chain.opentensor.ai:443"],
)
bis = mm.load_cache_block_info(bi_path="./exports/block_infos.json")
current_bh = sub.block_info().hash
netuids, data = mm.get_subnets_by(metric="emission", block_hash=current_bh, cutoff=30)
blocksnapshots = asyncio.run(
    mm.collect_blocksnapshots(
        block_info=bis[::-1][:5],  # NOTE: RCP exception for older blocks
        netuids=netuids[:5],
    )
)
mm.cache_blocksnapshots(blocksnapshots, outpath="./exports/bss0.json")
bss = mm.load_blocksnapshots(inpath="./exports/bss0.json")

print(len(blocksnapshots))
print("END")
