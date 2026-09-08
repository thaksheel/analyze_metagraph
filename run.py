import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import asyncio
from enum import Enum
import time
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, List, Optional, Literal, Dict

import bittensor
from bittensor import storage
from bittensor import Subtensor

# TODO: 1) use archive mode to fetch older blocks, 2) reduce runtime using asycn or query_batch, 3) organize and package all sn info


class StorageFunctions(Enum):
    """keys found in bittensor._generated.storage used in query_map functions"""

    # uids = "Uids" # do not need this 0-256 unless I need hotkeys
    dividends = "Dividends"
    last_update = "LastUpdate"
    active = "Active"
    # stake = "TotalStake" #TODO: find a way to collect stake, this throws an exception rn
    weight = "Weights"
    emission = "Emission"
    validator_permit = "ValidatorPermit"
    incentive = "Incentive"
    validator_trust = "ValidatorTrust"
    bonds = "Bonds"
    consensus = "Consensus"
    tempo = "Tempo"
    kappa = "Kappa"
    subnet_mechanism = "SubnetMechanism"
    rho = "Rho"
    activity_cutoff = "ActivityCutoff"
    mechanism_count_current = "MechanismCountCurrent"
    neuron_cert = "NeuronCertificates"


class BlockSnapshot:
    netuid: Any = []
    dividends: Any = []
    last_update: Any = []
    active: Any = []
    weight: Any = []
    emission: Any = []
    validator_permit: Any = []
    incentive: Any = []
    validator_trust: Any = []
    bonds: Any = []
    consensus: Any = []
    tempo: Any = []
    kappa: Any = []
    subnet_mechanism: Any = []
    rho: Any = []
    activity_cutoff: Any = []
    mechanism_count_current: Any = []
    neuron_cert: Any = []


async def main(sf: str, block_hash: int, block_snapshot: BlockSnapshot):
    client = bittensor.Client(network="finney")
    await client.connect()
    subs = client._substrate
    try:
        value = await subs.query_map(
            module="SubtensorModule",
            storage_function=sf,
            block_hash=block_hash,
            # param_sets=[[0]], # can be 0,1,2 
        )
        block_snapshot.netuid = np.array([v[0] for v in value])
        d = []
        for val in value:
            d.append(np.array(val[1]))
        setattr(block_snapshot, sf.lower(), d)
    except:
        print(f"---> {sf} exception")
        return None 
    return block_snapshot


sns = [
    17,
    46,
    128,
    65,
    31,
    122,
    61,
    110,
    119,
    117,
    127,
    120,
    54,
    104,
    92,
    62,
    13,
    103,
    1,
    6,
]  # len(sns)=20
blocks = [
    6837311,
    6880511,
    6923711,
    6966911,
    7010111,
    7053311,
    7096511,
    7139711,
    7182911,
    7226111,
    7269311,
    7312511,
    7355711,
    7398911,
    7442111,
    7485311,
    7528511,
    7571711,
    7614911,
    7658111,
    7701311,
    7744511,
    7787711,
    7830911,
    7874111,
    7917311,
]  # len(blocks) = 26
b = blocks[10]
durations = [time.time()]
stm = storage.SubtensorModule()
sub = Subtensor(
    network="finney",
    archive_endpoints=["wss://archive.chain.opentensor.ai:443"],
)
values = []
errors = []
worked = []
block_data: List[Dict] = []
for b in tqdm(blocks):
    print()
    bi = sub.block_info(b)
    block_d = BlockSnapshot()
    for sf in StorageFunctions:
        block_d = asyncio.run(main(sf=sf.value, block_hash=bi.hash, block_snapshot=block_d))
        if block_d is None:
            errors.append(sf)
        durations.append(time.time())
        print(f"fetch_time for {sf.value}= {durations[-1] - durations[-2]:.2f}s")
    block_data.append(
        {
            "block_num": bi.number,
            "block_timestamp": bi.timestamp,
            "block_snapshot": block_d,
        }
    )
    print(f"\n\nblock_num={bi.number}")

print(f"Runtime={(time.time() - durations[0]):.2f}s for len_blocks={len(blocks)}")
print("END")


# NOTE: explore scripts
active_sn10 = np.array(values[3][10][1])  # (fields, subnet number, select data)
active_sn10_uids = np.where(active_sn10 == True)

div = np.array(values[1][10][1])
div_active_sn10 = div[active_sn10_uids]
