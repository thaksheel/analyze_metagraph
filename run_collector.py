import asyncio
from bittensor import Subtensor

from modules.collector.metagraph_fetch import MetagraphManager

BLOCK_INFO = "./exports/block_infos10_test.json"
BLOCK_DIR = "./exports/blocks/"

mm = MetagraphManager(
    display=False,
    network="finney",
    client_kwargs={},
    max_concurrency=1,  # lower this if see RPC policy / rate-limit errors
)
# sub = Subtensor(
#     network="finney",
#     archive_endpoints=["wss://archive.chain.opentensor.ai:443"],
# )
# current_block = sub.block_info().number
# bis = asyncio.run(
#     mm.cache_block_info_with_retry(
#         sub=sub,
#         block_amount=10,
#         current_block=current_block,
#         duration_month=1,
#         save_path=BLOCK_INFO,
#     )
# )
bis = mm.load_cache_block_info(BLOCK_INFO)
netuids, scores = mm.get_subnets_by(
    metric="subnet_tao", block_hash=bis[-1].hash, cutoff=60
)

bss = asyncio.run(
    mm.collect_blocksnapshots_with_retry(
        block_info=bis, netuids=netuids, outpath=BLOCK_DIR
    )
)
# coldkeys + axon IPs are collected every block
mm.export_identity(BLOCK_DIR, "./exports/identity.json", n_points=5)

# single-file layout bt_analysis reads (sparse matrices, streamed block by block)
n = mm.export_for_analysis(BLOCK_DIR, "./exports/bss500_v2.json")
print(f"exported {n} blocks")

# example: square weights for one subnet at the first block
snap = mm.load_blocksnapshots(BLOCK_DIR, blocks=[bis[0].number], netuids=[netuids[0]])[
    0
]
sn = snap.subnets[netuids[0]]
W = sn.weight_matrix(mech=0, normalize=True)  # [n_uids, n_uids]; rows = validators
v_uids, W_v = sn.validator_by_miner()  # validator rows only
print(
    netuids[0],
    W.shape,
    len(v_uids),
    "validators; stake (alpha) of first:",
    sn.stake[v_uids[0]] / 1e9,
)
print("END")
