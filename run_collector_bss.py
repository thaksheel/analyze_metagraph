import asyncio
from bittensor import Subtensor
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from modules.collector.metagraph_fetch import MetagraphManager

BLOCK_INFO = "./exports/block_infos500_w43.json"
BLOCK_DIR = "./exports/blocks/"

mm = MetagraphManager(
    display=False,
    network="finney",
)
bis = mm.load_cache_block_info(BLOCK_INFO)
netuids, scores = mm.get_subnets_by(
    metric="subnet_tao", block_hash=bis[-1].hash, cutoff=60
)
snaps = mm.load_blocksnapshots(BLOCK_DIR, blocks=None)
nu = netuids[10]
sn = snaps[0]
W = sn.weights(netuid=nu, normalize=True)
v_uids, W_v = sn.validator_by_miner(netuid=nu)  # validator rows only
print(
    netuids[0],
    W.shape,
    len(v_uids),
    "validators; stake (alpha) of first:",
    # sn.stake[v_uids[0]] / 1e9,
)
print("END")
