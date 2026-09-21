import time
from bittensor import Subtensor

from src.metagraph_fetch import MetagraphManager

durations = [time.time()]
mm = MetagraphManager(display=False)
sub = Subtensor(
    network="finney",
    archive_endpoints=["wss://archive.chain.opentensor.ai:443"],
)
bis = mm.load_cache_block_info(bi_path="./exports/block_infos.json")
bss = mm.load_blocksnapshots(inpath="./exports/bss1.json")

print(len(bss))
print("END")
