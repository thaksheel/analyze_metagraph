import numpy as np 
from dataclasses import dataclass


@dataclass
class BlockSnapshot:
    netuid: int
    block_num: int
    consensus: int
    validator_trust: np.ndarray
    incentive: np.ndarray
    dividends: np.ndarray
    emissions: np.ndarray
    active: np.ndarray
    weights: np.ndarray
    bonds: np.ndarray
    active: np.ndarray
    validator_permit: np.ndarray
    uids: np.ndarray
    neurons: np.ndarray
    alpha_stake: np.ndarray
    stake: np.ndarray
    tao_stake: np.ndarray
    timestamp: float
    validator_trust: np.ndarray
    validator_permit: np.ndarray
    mechanism_count: int
    mechanism_id: int
    last_update: float
    hparams: float = None
