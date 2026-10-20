"""All settings, grouped by the analysis that uses them.

    cfg = Config()
    cfg.exploits.capture_min_share = 0.2
    cfg.search("window")        # every setting whose name or note mentions "window"
"""
from dataclasses import dataclass, field, fields
from typing import Dict, List

import pandas as pd

SECONDS_PER_BLOCK = 12
RAO = 1e9
U16 = 65535.0


def _f(default, note: str):
    """Dataclass field with a note (shown by Config.search / Config.describe)."""
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda d=default: type(d)(d), metadata={"note": note})
    return field(default=default, metadata={"note": note})


@dataclass
class PathConfig:
    blocks: str = _f("./exports/blocks/", "collector output: a directory of block_*.json.gz, or one merged .json")
    cache_dir: str = _f("./analysis_data/cache", "per-subnet arrays built from the blocks (reused on reruns)")
    results_dir: str = _f("./analysis_data/results", "CSV tables and summary.md")


@dataclass
class GeneralConfig:
    min_miner_share_factor: float = _f(0.1, "a miner matters if its mean share >= factor / n_miners")


@dataclass
class ExploitConfig:
    cp_min_segment_days: float = _f(2.0, "minimum days between change points")
    cp_penalty_scale: float = _f(3.0, "change-point penalty = scale * dims * log(T)")
    cp_min_effect: float = _f(1.0, "minimum standardized jump between segments")
    incident_window_days: float = _f(3.0, "before/after window around an incident")
    capture_min_share: float = _f(0.15, "capture event: a cohort gains >= this share of miner emission")
    cohort_capture_fraction: float = _f(0.5, "cohort = smallest set of gainers covering this share of gains")
    cohort_max_fraction: float = _f(0.2, "cohort never larger than this share of miners")
    recovery_unwind: float = _f(0.75, "recovered once this fraction of the cohort's excess is gone")
    recovery_consecutive: int = _f(3, "snapshots in a row needed to call recovery")
    burn_rise_threshold: float = _f(0.05, "rise in the owner/burn uid's incentive share that counts as a burn")


@dataclass
class TelemetryConfig:
    window_days: float = _f(14.0, "window for the distinguishability test")
    duplicate_similarity: float = _f(0.99, "profile cosine at/above this = indistinguishable miners")


@dataclass
class ScoringConfig:
    contested_cv: float = _f(1.0, "miner contested if CV of received weight across validators > this")
    copy_similarity: float = _f(0.99, "validator weights ~ previous consensus at/above this = copy-like")
    copy_persistence: float = _f(0.8, "copy-like only if it holds in this share of the last 2 days")
    stale_tempos: float = _f(2.0, "validator stale if (block - LastUpdate) > this * tempo")
    low_agreement: float = _f(0.7, "subnet has low agreement if mean pairwise cosine is below this")
    outcome_days: float = _f(7.0, "horizon for 'was a contested miner replaced?'")


@dataclass
class InsightConfig:
    signal_z: float = _f(3.0, "|z| vs trailing baseline that counts as a signal firing")
    baseline_days: float = _f(5.0, "trailing baseline length")
    lookback_days: float = _f(3.0, "how far before an incident a firing counts as a lead")
    variance_target: float = _f(0.9, "PCA: components needed to explain this much variance")
    top_signals: int = _f(3, "signals combined for the detection-gap estimate")


@dataclass
class SybilConfig:
    window_days: float = _f(7.0, "sliding window length")
    step_days: float = _f(1.0, "slide step; communities are re-detected each step")
    min_presence: float = _f(0.6, "miner present for >= this share of a window")
    event_z: float = _f(3.0, "|change in log share| >= z * robust sd counts as an event")
    support_change: float = _f(0.2, "1 - cosine of received-weight vector >= this counts as an event")
    common_event_frac: float = _f(0.3, "drop event times shared by >= this share of miners")
    sync_tolerance: int = _f(1, "snapshots of slack when matching events")
    min_events: int = _f(2, "events a miner needs for the synchrony layer")
    coreg_blocks: int = _f(360, "registrations within this many blocks = co-registered")
    layer_weights: Dict[str, float] = _f({"trajectory": 1.0, "emission_change": 1.0, "synchrony": 1.0,
                                          "validator_support": 1.0}, "weight of each similarity layer")
    layer_strong: float = _f(0.8, "a layer agrees on a pair at/above this similarity")
    min_layers: int = _f(2, "links need at least this many agreeing layers")
    edge_min: float = _f(0.6, "and a combined similarity at/above this")
    resolution: float = _f(1.0, "Louvain resolution")
    min_size: int = _f(3, "smallest community kept")
    min_density: float = _f(0.5, "share of possible internal links present")
    null_draws: int = _f(200, "random same-size groups in the significance test")
    null_p: float = _f(0.05, "significance level")
    match_jaccard: float = _f(0.3, "link communities across windows at this member overlap")
    min_track_windows: int = _f(3, "windows a track must last to be sybil-like")
    share_threshold: float = _f(0.15, "hypothesis threshold for clustered miners per subnet")


@dataclass
class RewardConfig:
    spike_baseline_days: float = _f(3.0, "trailing baseline for each miner")
    spike_z: float = _f(3.0, "spike: share >= z sd above baseline")
    spike_min_gain_factor: float = _f(1.0, "and gain >= factor / n_miners")
    spike_sd_floor_factor: float = _f(0.1, "baseline sd floor = factor / n_miners")
    support_ratio: float = _f(0.5, "unsupported if median-support gain < ratio * emission gain")
    repeat_min: int = _f(2, "unsupported spikes needed for a repeat offender")
    repeat_gap_days: float = _f(1.0, "spikes closer than this count once")
    systematic_p: float = _f(0.01, "binomial p-value for a systematic offender")
    churn_top_k: int = _f(10, "top-k set used for turnover")
    mech_burn: float = _f(0.5, "median burn share >= this -> burn_dominated")
    mech_wta_top1: float = _f(0.5, "median top-1 miner share >= this -> winner_take_all")
    mech_concentrated_top10: float = _f(0.7, "median top-10 share >= this -> concentrated")


@dataclass
class SufficiencyConfig:
    horizon_days: float = _f(1.0, "predict emission / rank this far ahead")
    coord_days: float = _f(2.0, "window for the future-coordination target")
    history_days: float = _f(2.0, "trailing window for volatility / synchrony features")
    train_frac: float = _f(0.7, "time split: first part trains, rest tests (with an embargo)")
    max_rows_per_subnet: int = _f(15000, "rows sampled per subnet")
    max_train_rows: int = _f(300000, "rows used to fit")
    r2_insufficient: float = _f(0.1, "R^2 below this -> signals insufficient")
    r2_adequate: float = _f(0.3, "R^2 at/above this -> adequate")


@dataclass
class RankingConfig:
    size_bands: List[float] = _f([0.05, 0.15, 0.30, 0.50], "share-of-subnets edges for size scores 1..5")
    impact_bands_pct: List[float] = _f([0.1, 0.5, 1.0, 3.0], "% of emission edges for impact scores 1..5")


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)
    exploits: ExploitConfig = field(default_factory=ExploitConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    insights: InsightConfig = field(default_factory=InsightConfig)
    sybil: SybilConfig = field(default_factory=SybilConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    sufficiency: SufficiencyConfig = field(default_factory=SufficiencyConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    seed: int = 0

    def describe(self) -> pd.DataFrame:
        rows = []
        for sec in fields(self):
            obj = getattr(self, sec.name)
            if not hasattr(obj, "__dataclass_fields__"):
                rows.append({"setting": sec.name, "value": obj, "note": ""})
                continue
            for f in fields(obj):
                rows.append({"setting": f"{sec.name}.{f.name}", "value": getattr(obj, f.name),
                             "note": f.metadata.get("note", "")})
        return pd.DataFrame(rows)

    def search(self, term: str) -> pd.DataFrame:
        d = self.describe()
        t = term.lower()
        return d[d["setting"].str.lower().str.contains(t) | d["note"].str.lower().str.contains(t)]
