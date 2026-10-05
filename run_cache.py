"""Caching required data points that adds runtime such as block information with number, hash, timestamps."""

from bittensor import Subtensor
import asyncio

from src.metagraph_fetch import MetagraphManager

mm = MetagraphManager(display=True)
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
        duration_month=4,
        save_path="./exports/block_infos500_4.json",
    )

# TODO: this is not working and goes into an infinit loop
# bis = asyncio.run(
#     mm.cache_block_info_with_retry(
#         sub,
#         block_amount=50,
#         current_block=current_block,
#         duration_month=1,
#         save_path="./exports/block_infos50.json",
#     )
# )
print("end")
