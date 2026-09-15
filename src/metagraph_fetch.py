import numpy as np
import pandas as pd
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Literal
from matplotlib import pyplot as plt
import bittensor
from datetime import datetime
from bittensor.metagraph import Metagraph, MetagraphNeuron
from bittensor import Substrate
from bittensor import Subtensor

from .utils import BlockSnapshot, BlockInfo

duration = [time.time()]


class MetagraphManager:
    def __init__(self, display:bool=False):
        self.display = display

    def block_collection(
        self,
        block_amount: int,
        current_block: int,
        duration_month: int,
    ):
        # NOTE: 12s/block
        blocks_per_amount = int(duration_month * (30 * 24 * 3600) / 12 / block_amount)
        blocks = [
            current_block - i * blocks_per_amount for i in range(block_amount, -1, -1)
        ]
        return blocks

    def cache_block_info(
        self,
        sub: Subtensor,
        block_amount: int,
        current_block: int,
        duration_month: int,
        save_path: str,
    ):
        blocks = self.block_collection(block_amount, current_block, duration_month)
        bis = [sub.block_info(b).__dict__ for b in blocks]
        bis = []
        for b in blocks:
            bi = sub.block_info(b).__dict__
            bi['timestamp'] = bi['timestamp'].isoformat()
            bis.append(bi)
        with open(save_path, "w") as f: 
            json.dump(bis, f)
        return bis 

    def load_cache_block_info(self, bi_path: str) -> List[BlockInfo]: 
        """Collects only the hash, number, and timestamps as `BlockInfo` fields while ignoring the rest."""
        with open(bi_path, "r") as f:
            data = json.load(f)
        blocks = []
        for entry in data:
            blocks.append(
                BlockInfo(
                    number=entry["number"],
                    hash=entry["hash"],
                    timestamp=datetime.fromisoformat(entry["timestamp"])
                )
            )
        return blocks 


def load_snapshot_at_block(m: Metagraph, sub: Subtensor, hparams: float, block: int):
    m.sync(subtensor=sub, block=block)
    sd = m.state_dict()
    return BlockSnapshot(
        netuid=sd["netuid"],
        block_num=block,
        consensus=sd["consensus"],
        validator_trust=sd["validator_trust"],
        incentive=sd["incentive"],
        dividends=sd["dividends"],
        emissions=sd["emission"],
        active=sd["active"],
        weights=sd["weights"],
        bonds=sd["bonds"],
        validator_permit=sd["validator_permit"],
        uids=sd["uids"],
        neurons=sd["neurons"],
        alpha_stake=sd["alpha_stake"],
        stake=sd["stake"],
        tao_stake=sd["tao_stake"],
        timestamp=sub.get_block_info(block).timestamp,
        mechanism_count=m.mechanism_count,
        mechanism_id=m.mechid,
        last_update=sd["last_update"],
        hparams=hparams,
    )


def to_json(snapshots: List[BlockSnapshot], filename: str, path: str):
    data = []
    snapshots = [snap.__dict__ for snap in snapshots]
    keys = list(snapshots[-1].keys())
    for snap in snapshots:
        d = dict(zip(keys, [None for _ in keys]))
        for key in keys:
            if key == "hparams":
                d[key] = snap[key].__dict__
                continue
            elif key == "neurons":
                continue
            d[key] = (
                snap[key].tolist() if isinstance(snap[key], np.ndarray) else snap[key]
            )
        data.append(d)
    with open(path + filename, "w") as f:
        json.dump(data, f, indent=4)
        print("exported", len(data), len(data[0]["active"]))
    return data


def block_collection(
    block_amount: int,
    current_block: int,
    duration_month: int,
):
    # NOTE: 12s/block
    blocks_per_amount = int(duration_month * (30 * 24 * 3600) / 12 / block_amount)
    blocks = [
        current_block - i * blocks_per_amount for i in range(block_amount, -1, -1)
    ]
    return blocks


def get_metagraph(
    netuid: int, block_amount: int, duration_month: int, display: bool = False
):
    m = Metagraph(
        netuid=netuid,
        network="finney",
        lite=False,
        sync=True,
    )
    subtensor = Subtensor("archive")
    blocks = block_collection(
        block_amount=block_amount,
        current_block=subtensor.block,
        duration_month=duration_month,
    )
    snapshots: List[BlockSnapshot] = []
    for i, block in enumerate(blocks):
        if display:
            if i % 10 == 0 and i > 0:
                active = snapshots[-1].active
                duration.append(time.time())
                dur = time.time() - duration[-1]
                print(
                    f"progress={round(100*i/len(blocks),2)}% block_count={i}, miner_count={active.sum()}, duration={round(dur/60,2)}mins"
                )
        snapshots.append(
            load_snapshot_at_block(m=m, sub=subtensor, block=block, hparams=None)
        )
    if display:
        keys = [
            "emissions",
            "dividends",
            "active",
            "weights",
            "stake",
            "incentive",
            "validator_trust",
            "uids",
        ]
        for key in keys:
            d = snapshots[0].__dict__[key][snapshots[0].__dict__["active"] == 1]
            print(key, d.shape, d.sum())
            print(d)
            print()
    return snapshots


def snapshot_to_df(snapshots: List[BlockSnapshot]):
    records = []
    for snap in snapshots:
        active_count = snap.active.sum()
        active_mask = snap.active == 1
        records.append(
            {
                "block": snap.block_num,
                "active_count": active_count,
                "total_incentive": snap.incentive[active_mask].sum(),
                "total_emissions": snap.emissions[active_mask].sum(),
                "total_dividends": snap.dividends[active_mask].sum(),
                "total_weights": snap.weights[active_mask].sum(),
                "total_val_trust": snap.validator_trust[active_mask].sum(),
                "sum_active_uids": snap.uids[active_mask].sum(),
                "total_vals_stake(mil)": snap.tao_stake.sum() / 1e6,
                "timestamp": pd.to_datetime(snap.timestamp, unit="ms"),
            }
        )
    return pd.DataFrame(records)
