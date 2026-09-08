import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import asyncio
from enum import Enum
import time
from tqdm import tqdm

import bittensor
from bittensor.metagraph import Metagraph, MetagraphNeuron
from bittensor import storage
from bittensor import Subtensor

# TODO: 1) use archive mode to fetch older blocks, 2) reduce runtime using asycn or query_batch, 3) organize and package all sn info

class StorageFunctions(Enum): 
    """keys found in bittensor._generated.storage used in query_map functions"""
    uids = "Uids"
    dividends = "Dividends"
    last_update = "LastUpdate"
    active = "Active"
    stake = "TotalStake"
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


async def main(sf: str, block_hash: int):
    client = bittensor.Client(network="finney")
    await client.connect()
    subs = client._substrate
    try:
        value = await subs.query_map(
            module="SubtensorModule",
            storage_function=sf,
            block_hash=block_hash,
            # params=[33],  # netuid, or uid
        )
    except:
        print(f"---> {sf} exception")
        return None
    return value


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
sub = Subtensor(network="finney")
values = []
errors = []
worked = []
for b in tqdm(blocks):
    # b = sub.block_info(b).hash
    b = None 
    for sf in StorageFunctions:
        val = asyncio.run(main(sf=sf.value, block_hash=b))
        if val is None:
            errors.append(sf)
        else:
            values.append(val)
            worked.append(sf.value)
        durations.append(time.time())
        print(f"fetch_time for {sf.value} ={durations[-1] - durations[-2]:.2f}s")
    print(f"\n\nblock_num={b}")

print(f"Runtime={(time.time() - durations[0]):.2f}s for len_blocks={len(blocks)}")
print("END")
