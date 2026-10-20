"""Central configuration.

Every threshold used by the analyses lives here so results are reproducible and
easy to tune. Field names map canonical names used in this codebase to the keys
that appear in your raw snapshot JSON (the storage function names you query).
Check the storage names against your subtensor version and edit RAW_FIELDS if
they differ.
"""
from dataclasses import dataclass, field
from typing import Dict, List

# canonical name -> key in raw snapshot JSON
RAW_FIELDS: Dict[str, str] = {
    # per-UID vectors (already collected)
    "incentive": "Incentive",
    "emission": "Emission",
    "dividends": "Dividends",
    "consensus": "Consensus",
    "vtrust": "ValidatorTrust",
    "active": "Active",
    "vpermit": "ValidatorPermit",
    "last_update": "LastUpdate",
    # matrices (double maps)
    "weights": "Weights",
    "bonds": "Bonds",
    # miner-emitted
    "certificates": "NeuronCertificates",
    # subnet-level scalars (already collected)
    "kappa": "Kappa",
    "rho": "Rho",
    "tempo": "Tempo",
    "activity_cutoff": "ActivityCutoff",
    "mechanism": "SubnetMechanism",
    "mechanism_count": "MechanismCountCurrent",
    # suggested additions, collected every snapshot
    "hotkeys": "Keys",
    "reg_block": "BlockAtRegistration",
    "subnet_tao": "SubnetTAO",
    "alpha_in": "SubnetAlphaIn",
    "owner_hotkey": "SubnetOwnerHotkey",
    "yc3": "Yuma3On",
    # collected by the updated collector (codegen storage items)
    "coldkeys": "Coldkeys",                 # Owner(hotkey) per uid
    "axon_ip": "AxonIP",                    # Axons(netuid, hotkey).ip per uid
    "axon_block": "AxonBlock",              # block of the uid's last serve_axon
    "prometheus_block": "PrometheusBlock",
    "commit_block": "CommitmentBlock",      # Commitments.CommitmentOf(netuid, hotkey).block
    "stake": "TotalHotkeyAlpha",            # alpha staked to each uid's hotkey on this subnet
    "stake_weight": "StakeWeight",
    "commit_reveal": "CommitRevealWeightsEnabled",
    "moving_price": "SubnetMovingPrice",
}

HYPERPARAMS: List[str] = [
    "kappa", "rho", "tempo", "activity_cutoff", "mechanism", "mechanism_count",
]

RAO_PER_TAO = 1e9
SECONDS_PER_BLOCK = 12


@dataclass
class Config:
    # ---- paths ----
    raw_snapshots: str = "./exports/bss500.json"
    identity_file: str = ""  # optional: Owner + Axons at a few snapshots
    processed_dir: str = "./analysis_data/processed"
    results_dir: str = "./analysis_data/results"
    compress_processed: bool = False     # True: ~5x smaller .npz files, slower prepare

    # ---- general ----
    min_miner_share_factor: float = 0.1  # miner "matters" if mean share >= factor / n_miners
    window_days: float = 14.0            # window for clustering / distinguishability

    # ---- Area 1: exploit cycles ----
    cp_min_segment_days: float = 2.0     # minimum days between change points
    cp_penalty_scale: float = 3.0        # penalty = scale * dims * log(T)
    cp_min_effect: float = 1.0           # min standardized jump between segments
    incident_window_days: float = 3.0    # before/after window around a change point
    cohort_capture_fraction: float = 0.5  # cohort = smallest set capturing this share of gains
    cohort_max_fraction: float = 0.2     # ...but never more than this share of UIDs
    capture_min_share: float = 0.15      # capture event: cohort gains >= this share of miner emission
    recovery_unwind: float = 0.75        # recovered once this fraction of the cohort's excess is gone
    recovery_consecutive: int = 3        # snapshots in a row needed to call recovery
    burn_rise_threshold: float = 0.05    # owner/burn incentive share rise that counts as a burn

    # ---- Area 2: telemetry ----
    duplicate_similarity: float = 0.99   # profile cosine at/above this = indistinguishable

    # ---- Area 3: MVS ----
    contested_cv: float = 1.0            # miner contested if CV of received weights > this
    copy_similarity: float = 0.99        # validator weights ~ lagged consensus => copy-like
    stale_tempos: float = 2.0            # stale if (block - last_update) > stale_tempos * tempo
    low_agreement: float = 0.7           # subnet "low agreement" if mean cosine below this

    # ---- Area 4: actionable insights ----
    signal_z: float = 3.0                # |z| vs trailing baseline that counts as a firing
    baseline_days: float = 5.0           # trailing baseline length
    lookback_days: float = 3.0           # how far before an incident a lead can count
    variance_target: float = 0.9         # PCA: components needed to explain this

    # ---- Area 5: sybil ----
    traj_corr: float = 0.95              # de-meaned log-share trajectory correlation for an edge
    weight_profile_corr: float = 0.9     # validator-preference profile correlation for an edge
    coreg_blocks: int = 360              # registrations within this many blocks = co-registered
    coreg_traj_corr: float = 0.85        # weaker trajectory bar if co-registered
    min_cluster_size: int = 3
    sybil_share_threshold: float = 0.15  # hypothesis threshold

    # ---- B1: dynamic multi-layer community detection (sybil-like clusters) ----
    dc_window_days: float = 7.0          # sliding window length
    dc_step_days: float = 1.0            # slide step (communities are re-detected each step)
    dc_min_presence: float = 0.6         # miner present for >= this share of the window
    dc_event_z: float = 3.0              # |change in log share| >= z * robust sd counts as an event
    dc_support_change: float = 0.2       # 1 - cosine of received-weight vector >= this is an event
    dc_common_event_frac: float = 0.3    # drop event times shared by >= this share of miners
    dc_sync_tolerance: int = 1           # snapshots of slack when matching events
    dc_min_events: int = 2               # miners need this many events for the sync layer
    dc_layer_weights: Dict[str, float] = field(default_factory=lambda: {
        "trajectory": 1.0, "emission_change": 1.0, "synchrony": 1.0, "validator_support": 1.0})
    dc_layer_strong: float = 0.8         # a layer "agrees" on a pair at/above this similarity
    dc_min_layers: int = 2               # edges need >= this many agreeing layers
    dc_edge_min: float = 0.6             # and combined similarity >= this
    dc_resolution: float = 1.0           # Louvain resolution
    dc_min_size: int = 3
    dc_min_density: float = 0.5          # share of possible internal edges present
    dc_null_draws: int = 200             # random same-size groups for the significance test
    dc_null_p: float = 0.05
    dc_match_jaccard: float = 0.3        # link communities across windows at this member overlap
    dc_min_track_windows: int = 3        # persistence needed to call a track sybil-like

    # ---- B2: abnormal rewards without consensus improvement ----
    spike_baseline_days: float = 3.0     # trailing baseline for each miner
    spike_z: float = 3.0
    spike_min_gain_factor: float = 1.0   # gain must be >= factor / n_miners (one fair share)
    spike_sd_floor_factor: float = 0.1   # baseline sd floor = factor / n_miners
    support_ratio: float = 0.5           # unsupported if support gain < ratio * emission gain
    repeat_min: int = 2                  # unsupported spikes needed to call a miner a repeat
    repeat_gap_days: float = 1.0         # spikes closer than this count once
    churn_top_k: int = 10
    mech_burn: float = 0.5               # median burn share >= this -> burn_dominated
    mech_wta_top1: float = 0.5           # median top-1 miner share >= this -> winner_take_all
    mech_concentrated_top10: float = 0.7  # median top-10 share >= this -> concentrated

    # ---- B3: are metagraph signals enough? (out-of-sample R^2) ----
    pred_horizon_days: float = 1.0       # predict emission / rank this far ahead
    pred_coord_days: float = 2.0         # window for the future-coordination target
    pred_history_days: float = 2.0       # trailing window for volatility / synchrony features
    pred_train_frac: float = 0.7         # time split: first 70% train, rest test (with embargo)
    pred_max_rows_per_subnet: int = 15000
    pred_max_train_rows: int = 300000
    r2_insufficient: float = 0.1         # R^2 below this -> signals insufficient
    r2_adequate: float = 0.3             # R^2 at/above this -> adequate
    seed: int = 0

    # ---- ranking (1..5 bands; value >= band edge moves up one score) ----
    size_bands: List[float] = field(default_factory=lambda: [0.05, 0.15, 0.30, 0.50])
    impact_bands_pct: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 3.0])

    def snapshots_per_day(self, block_spacing: float) -> float:
        return 86400.0 / (block_spacing * SECONDS_PER_BLOCK)
