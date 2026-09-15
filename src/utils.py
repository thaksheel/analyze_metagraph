import numpy as np 
from dataclasses import dataclass
from typing import List, Literal, Optional, Any 
from enum import Enum
from datetime import datetime


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

@dataclass 
class BlockInfo: 
    number: int
    hash: str 
    timestamp: datetime
