import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Optional, Any
from enum import Enum
from datetime import datetime
from numpy.typing import NDArray


class StorageFunctions(Enum):
    """keys found in bittensor._generated.storage used in query_map functions"""

    # uids = "Uids" # do not need this 0-256 unless I need hotkeys
    emission = "Emission"
    dividends = "Dividends"
    last_update = "LastUpdate"
    active = "Active"
    # stake = "TotalStake" #TODO: find a way to collect stake, this throws an exception rn
    weight = "Weights"
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


@dataclass
class BlockInfo:
    number: int
    hash: str
    timestamp: datetime


@dataclass
class BlockSnapshot:
    block_info: BlockInfo
    netuid: List[int]

    # Arrays indexed by netuid
    Active: List[NDArray] = None
    ActivityCutoff: List[NDArray] = None
    Bonds: List[NDArray] = None
    Consensus: List[NDArray] = None
    Dividends: List[NDArray] = None
    Emission: List[NDArray] = None
    Incentive: List[NDArray] = None
    Kappa: List[NDArray] = None
    LastUpdate: List[NDArray] = None
    MechanismCountCurrent: List[NDArray] = None
    NeuronCertificates: List[NDArray] = None
    Rho: List[NDArray] = None
    SubnetMechanism: List[NDArray] = None
    Tempo: List[NDArray] = None
    ValidatorPermit: List[NDArray] = None
    ValidatorTrust: List[NDArray] = None
    Weights: List[NDArray] = None
