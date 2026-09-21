import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional, Literal, Tuple
from matplotlib import pyplot as plt

from . import BlockSnapshot, BlockInfo


class MetagraphScanner:
    def __init__(self, seed: int = 42, display: bool = False):
        self.seed = seed
        self.display = display 

    def identify_neurons(self, ): 
        """find miners and validators from list of neurons ids"""
        pass 

    def aggr_metagraph_data(self, snapshots: List[BlockSnapshot], active_uids: List[int]):
        data = {}
        exclude = ['block_info', "netuid"]
        metagraph_fields = [k for k in BlockSnapshot().__dict__.keys() if k not in exclude]
        for key in metagraph_fields:
            col = f"tot_{key}"
            data[col] = [
                sum([s.__dict__[key][i] for s in snapshots])[0]
                for i, au in enumerate(active_uids)
            ]
        return data

    def aggregate_blocks(self, snapshots: List[BlockSnapshot]) -> pd.DataFrame:
        # TODO: now bss is block by uids by time so what I have here does not quite work 
        active_uid_arr = [np.where(s.Active[0] == 1)[0] for s in snapshots]
        unique_ids, counts = np.unique(
            np.concatenate(active_uid_arr), return_counts=True
        )
        df = pd.DataFrame(
            {
                "uids": unique_ids,
                "counts": counts,
                "netuid": snapshots[0].netuid,
            }
        )
        metagraph = self.aggr_metagraph_data(snapshots, unique_ids)
        for k, v in metagraph.items():
            df[k] = v
        return df

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
                    timestamp=datetime.fromisoformat(entry["timestamp"]),
                )
            )
        return blocks

    def load_blocksnapshots(self, inpath: str):
        def to_array_list(x):
            return [np.array(v) for v in x]

        with open(inpath, "r") as f:
            raw = json.load(f)
        snapshots = []
        for entry in raw:
            bi_dict = entry["block_info"]
            block_info = BlockInfo(**bi_dict)
            snapshot = BlockSnapshot(
                block_info=block_info,
                netuid=entry["netuid"],
                Active=to_array_list(entry["Active"]),
                ActivityCutoff=to_array_list(entry["ActivityCutoff"]),
                Bonds=to_array_list(entry["Bonds"]),
                Consensus=to_array_list(entry["Consensus"]),
                Dividends=to_array_list(entry["Dividends"]),
                Emission=to_array_list(entry["Emission"]),
                Incentive=to_array_list(entry["Incentive"]),
                Kappa=to_array_list(entry["Kappa"]),
                LastUpdate=to_array_list(entry["LastUpdate"]),
                MechanismCountCurrent=to_array_list(entry["MechanismCountCurrent"]),
                NeuronCertificates=to_array_list(entry["NeuronCertificates"]),
                Rho=to_array_list(entry["Rho"]),
                SubnetMechanism=to_array_list(entry["SubnetMechanism"]),
                Tempo=to_array_list(entry["Tempo"]),
                ValidatorPermit=to_array_list(entry["ValidatorPermit"]),
                ValidatorTrust=to_array_list(entry["ValidatorTrust"]),
                Weights=to_array_list(entry["Weights"]),
            )
            snapshots.append(snapshot)
        return snapshots
