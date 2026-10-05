import asyncio
import time
from bittensor import Subtensor

from src.metagraph_fetch import MetagraphManager

durations = [time.time()]
mm = MetagraphManager(display=False)
sub = Subtensor(
    network="finney",
    archive_endpoints=["wss://archive.chain.opentensor.ai:443"],
)
bis = mm.load_cache_block_info(bi_path="./exports/block_infos500.json")
current_bh = sub.block_info().hash 
netuids, data = mm.get_subnets_by(
    metric="emission",
    block_hash=current_bh,
    cutoff=60,
)
blocksnapshots = asyncio.run(
    mm.collect_blocksnapshots_with_retry(
        block_info=bis,
        netuids=netuids,
        outpath="./exports/checkpoint/",
        checkpoint=True, 
    )
)
mm.cache_blocksnapshots(blocksnapshots, outpath="./exports/bss500.json")
bss = mm.load_blocksnapshots(inpath="./exports/bss500.json")

print(len(blocksnapshots))
print("END")
