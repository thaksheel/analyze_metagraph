"""Storage item descriptors from the runtime codegen (spec version 445).

If the generated module is importable (bittensor._generated.storage, or a local
`storage.py` you regenerate with `python -m codegen <ws-endpoint>`), its items are used,
so names always match the runtime you generated against. Otherwise the vendored subset
below, copied from that file, is used.

An Item unpacks to (module, storage_function) and carries value_type_ident, which the
collector uses to decode values (PerU16, AlphaBalance, AxonInfo, AccountId32, ...).
Key layout is not part of the descriptor, so `utils.DEFAULT_SPECS` records how each
item is keyed.
"""
from typing import NamedTuple

GENERATED_SOURCE = None
try:  # pragma: no cover - depends on the installed bittensor
    from bittensor._generated.storage import Commitments, SubtensorModule  # type: ignore
    GENERATED_SOURCE = "bittensor._generated.storage"
except Exception:
    try:
        from storage import Commitments, SubtensorModule  # type: ignore  # locally regenerated file
        GENERATED_SOURCE = "storage"
    except Exception:
        GENERATED_SOURCE = None

if GENERATED_SOURCE is None:
    class Item(NamedTuple):
        """A (container, name) pair; unpack into query/constant calls."""
        container: str
        name: str
        value_type_ident: str = ''

        def __iter__(self):
            return iter((self.container, self.name))

    class SubtensorModule:  # vendored subset, spec version 445
        # sizing / mechanisms
        SubnetworkN = Item('SubtensorModule', 'SubnetworkN', 'u16')
        MechanismCountCurrent = Item('SubtensorModule', 'MechanismCountCurrent', 'MechId')
        MechanismEmissionSplit = Item('SubtensorModule', 'MechanismEmissionSplit', 'Vec<u16>')
        # per-uid vectors
        Active = Item('SubtensorModule', 'Active', 'Vec<bool>')
        Consensus = Item('SubtensorModule', 'Consensus', 'Vec<PerU16>')
        Incentive = Item('SubtensorModule', 'Incentive', 'Vec<PerU16>')
        Dividends = Item('SubtensorModule', 'Dividends', 'Vec<PerU16>')
        Emission = Item('SubtensorModule', 'Emission', 'Vec<AlphaBalance>')
        LastUpdate = Item('SubtensorModule', 'LastUpdate', 'Vec<u64>')
        ValidatorTrust = Item('SubtensorModule', 'ValidatorTrust', 'Vec<PerU16>')
        ValidatorPermit = Item('SubtensorModule', 'ValidatorPermit', 'Vec<bool>')
        StakeWeight = Item('SubtensorModule', 'StakeWeight', 'Vec<u16>')
        # double maps
        Weights = Item('SubtensorModule', 'Weights', 'Vec<(u16, u16)>')
        Bonds = Item('SubtensorModule', 'Bonds', 'Vec<(u16, u16)>')
        Keys = Item('SubtensorModule', 'Keys', 'AccountId32')
        BlockAtRegistration = Item('SubtensorModule', 'BlockAtRegistration', 'u64')
        Axons = Item('SubtensorModule', 'Axons', 'AxonInfo')
        Prometheus = Item('SubtensorModule', 'Prometheus', 'PrometheusInfo')
        NeuronCertificates = Item('SubtensorModule', 'NeuronCertificates', 'NeuronCertificate')
        # stake / ownership
        TotalHotkeyAlpha = Item('SubtensorModule', 'TotalHotkeyAlpha', 'AlphaBalance')
        ParentKeys = Item('SubtensorModule', 'ParentKeys', 'Vec<(u64, AccountId32)>')
        Owner = Item('SubtensorModule', 'Owner', 'AccountId32')
        # subnet economics
        SubnetTAO = Item('SubtensorModule', 'SubnetTAO', 'TaoBalance')
        SubnetAlphaIn = Item('SubtensorModule', 'SubnetAlphaIn', 'AlphaBalance')
        SubnetAlphaOut = Item('SubtensorModule', 'SubnetAlphaOut', 'AlphaBalance')
        SubnetVolume = Item('SubtensorModule', 'SubnetVolume', 'u128')
        SubnetMovingPrice = Item('SubtensorModule', 'SubnetMovingPrice', 'FixedI128')
        SubnetTaoInEmission = Item('SubtensorModule', 'SubnetTaoInEmission', 'TaoBalance')
        SubnetAlphaInEmission = Item('SubtensorModule', 'SubnetAlphaInEmission', 'AlphaBalance')
        SubnetAlphaOutEmission = Item('SubtensorModule', 'SubnetAlphaOutEmission', 'AlphaBalance')
        SubnetOwner = Item('SubtensorModule', 'SubnetOwner', 'AccountId32')
        SubnetOwnerHotkey = Item('SubtensorModule', 'SubnetOwnerHotkey', 'AccountId32')
        NetworkRegisteredAt = Item('SubtensorModule', 'NetworkRegisteredAt', 'u64')
        # hyperparameters
        SubnetMechanism = Item('SubtensorModule', 'SubnetMechanism', 'u16')
        Tempo = Item('SubtensorModule', 'Tempo', 'u16')
        Kappa = Item('SubtensorModule', 'Kappa', 'u16')
        Rho = Item('SubtensorModule', 'Rho', 'u16')
        ActivityCutoff = Item('SubtensorModule', 'ActivityCutoff', 'u16')
        ImmunityPeriod = Item('SubtensorModule', 'ImmunityPeriod', 'u16')
        MaxAllowedUids = Item('SubtensorModule', 'MaxAllowedUids', 'u16')
        MaxAllowedValidators = Item('SubtensorModule', 'MaxAllowedValidators', 'u16')
        MinAllowedWeights = Item('SubtensorModule', 'MinAllowedWeights', 'u16')
        MaxWeightsLimit = Item('SubtensorModule', 'MaxWeightsLimit', 'u16')
        WeightsVersionKey = Item('SubtensorModule', 'WeightsVersionKey', 'u64')
        WeightsSetRateLimit = Item('SubtensorModule', 'WeightsSetRateLimit', 'u64')
        BondsMovingAverage = Item('SubtensorModule', 'BondsMovingAverage', 'u64')
        BondsPenalty = Item('SubtensorModule', 'BondsPenalty', 'u16')
        CommitRevealWeightsEnabled = Item('SubtensorModule', 'CommitRevealWeightsEnabled', 'bool')
        LiquidAlphaOn = Item('SubtensorModule', 'LiquidAlphaOn', 'bool')
        Yuma3On = Item('SubtensorModule', 'Yuma3On', 'bool')
        AlphaValues = Item('SubtensorModule', 'AlphaValues', '(u16, u16)')
        Burn = Item('SubtensorModule', 'Burn', 'TaoBalance')
        Difficulty = Item('SubtensorModule', 'Difficulty', 'u64')
        RegistrationsThisInterval = Item('SubtensorModule', 'RegistrationsThisInterval', 'u16')
        NetworkRegistrationAllowed = Item('SubtensorModule', 'NetworkRegistrationAllowed', 'bool')
        # network-wide values
        TotalStake = Item('SubtensorModule', 'TotalStake', 'TaoBalance')
        TotalIssuance = Item('SubtensorModule', 'TotalIssuance', 'TaoBalance')
        BlockEmission = Item('SubtensorModule', 'BlockEmission', 'u64')
        TaoWeight = Item('SubtensorModule', 'TaoWeight', 'u64')

    class Commitments:
        CommitmentOf = Item('Commitments', 'CommitmentOf', 'Registration')
else:  # pragma: no cover
    Item = type(SubtensorModule.Keys)
