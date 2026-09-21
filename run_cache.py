"""Caching required data points that adds runtime such as block information with number, hash, timestamps."""
from bittensor import Subtensor

from src.metagraph_fetch import MetagraphManager

mm = MetagraphManager()
sub = Subtensor(
    network="finney",
    archive_endpoints=["wss://archive.chain.opentensor.ai:443"],
)

# NOTE: caching block info with number, hash, timestamps 
current_block = sub.block_info().number
bis = mm.cache_block_info(
    sub,
    block_amount=500,
    current_block=current_block,
    duration_month=2,
    save_path="./exports/block_infos500.json",
)

