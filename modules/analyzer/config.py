"""All settings, grouped by the analysis that uses them.

cfg = Config()
cfg.exploits.capture_min_share = 0.2
cfg.search("window")        # every setting whose name or note mentions "window"
"""

import inspect
import re
from dataclasses import dataclass, field, fields
from typing import Dict, List

import pandas as pd

SECONDS_PER_BLOCK = 12
RAO = 1e9
U16 = 65535.0


@dataclass
class PathConfig:
    """Where data is read from and written to.

    Attributes
    ----------
    blocks : str
        Collector output: a directory of ``block_*.json.gz`` files, or one merged ``.json``.
    cache_dir : str
        Per-subnet arrays built from the blocks, reused on reruns.
    results_dir : str
        Directory for CSV tables and ``summary.md``.
    """

    blocks: str = "./exports/blocks/"
    cache_dir: str = "./analysis_data/cache"
    results_dir: str = "./analysis_data/results"


@dataclass
class GeneralConfig:
    """Settings shared across analyses.

    Attributes
    ----------
    min_miner_share_factor : float
        A miner matters if its mean share is at least ``factor / n_miners``.
    """

    min_miner_share_factor: float = 0.1


@dataclass
class ExploitConfig:
    """Settings for detecting exploits: change points, incidents, capture events, and recovery.

    Attributes
    ----------
    cp_min_segment_days : float
        Minimum number of days between change points.
    cp_penalty_scale : float
        Change-point penalty is ``scale * dims * log(T)``.
    cp_min_effect : float
        Minimum standardized jump between segments.
    incident_window_days : float
        Before/after window around an incident.
    capture_min_share : float
        Capture event: a cohort gains at least this share of miner emission.
    cohort_capture_fraction : float
        A cohort is the smallest set of gainers covering this share of gains.
    cohort_max_fraction : float
        A cohort is never larger than this share of miners.
    recovery_unwind : float
        Recovered once this fraction of the cohort's excess is gone.
    recovery_consecutive : int
        Snapshots in a row needed to call recovery.
    burn_rise_threshold : float
        Rise in the owner/burn UID's incentive share that counts as a burn.
    """

    cp_min_segment_days: float = 2.0
    cp_penalty_scale: float = 3.0
    cp_min_effect: float = 1.0
    incident_window_days: float = 3.0
    capture_min_share: float = 0.15
    cohort_capture_fraction: float = 0.5
    cohort_max_fraction: float = 0.2
    recovery_unwind: float = 0.75
    recovery_consecutive: int = 3
    burn_rise_threshold: float = 0.05


@dataclass
class TelemetryConfig:
    """Settings for checking whether miners can be told apart from their telemetry.

    Attributes
    ----------
    window_days : float
        Window length for the distinguishability test.
    duplicate_similarity : float
        Profile cosine at or above this value means indistinguishable miners.
    """

    window_days: float = 14.0
    duplicate_similarity: float = 0.99


@dataclass
class ScoringConfig:
    """Settings for judging validator behavior and weight-setting quality.

    Attributes
    ----------
    contested_cv : float
        A miner is contested if the CV of its received weight across validators exceeds this.
    copy_similarity : float
        Validator weights at or above this similarity to the previous consensus are copy-like.
    copy_persistence : float
        Copy-like only if it holds in this share of the last 2 days.
    stale_tempos : float
        A validator is stale if ``(block - LastUpdate) > stale_tempos * tempo``.
    low_agreement : float
        A subnet has low agreement if its mean pairwise cosine is below this.
    outcome_days : float
        Horizon for asking whether a contested miner was replaced.
    """

    contested_cv: float = 1.0
    copy_similarity: float = 0.99
    copy_persistence: float = 0.8
    stale_tempos: float = 2.0
    low_agreement: float = 0.7
    outcome_days: float = 7.0


@dataclass
class InsightConfig:
    """Settings for early-warning signals that lead incidents.

    Attributes
    ----------
    signal_z : float
        Absolute z-score against the trailing baseline that counts as a signal firing.
    baseline_days : float
        Length of the trailing baseline.
    lookback_days : float
        How far before an incident a firing counts as a lead.
    variance_target : float
        PCA: fraction of variance the components must explain.
    top_signals : int
        Number of signals combined for the detection-gap estimate.
    """

    signal_z: float = 3.0
    baseline_days: float = 5.0
    lookback_days: float = 3.0
    variance_target: float = 0.9
    top_signals: int = 3


@dataclass
class SybilConfig:
    """Settings for detecting clusters of miners that behave like one operator (sybils).

    Attributes
    ----------
    window_days : float
        Sliding window length.
    step_days : float
        Slide step. Communities are re-detected at each step.
    min_presence : float
        Miner must be present for at least this share of a window.
    event_z : float
        An absolute change in log share of at least ``event_z * robust_sd`` counts as an event.
    support_change : float
        An event occurs when ``1 - cosine`` of the received-weight vector is at least this.
    common_event_frac : float
        Event times shared by at least this share of miners are dropped.
    sync_tolerance : int
        Snapshots of slack when matching events.
    min_events : int
        Events a miner needs to take part in the synchrony layer.
    coreg_blocks : int
        Registrations within this many blocks count as co-registered.
    layer_weights : dict of str to float
        Weight of each similarity layer: trajectory, emission_change, synchrony, validator_support.
    layer_strong : float
        A layer agrees on a pair when similarity is at or above this.
    min_layers : int
        Links need at least this many agreeing layers.
    edge_min : float
        Links also need a combined similarity at or above this.
    resolution : float
        Louvain resolution parameter.
    min_size : int
        Smallest community kept.
    min_density : float
        Share of possible internal links that must be present.
    null_draws : int
        Number of random same-size groups in the significance test.
    null_p : float
        Significance level.
    match_jaccard : float
        Member overlap needed to link communities across windows.
    min_track_windows : int
        Number of windows a track must last to be sybil-like.
    share_threshold : float
        Hypothesis threshold for the share of clustered miners per subnet.
    """

    window_days: float = 7.0
    step_days: float = 1.0
    min_presence: float = 0.6
    event_z: float = 3.0
    support_change: float = 0.2
    common_event_frac: float = 0.3
    sync_tolerance: int = 1
    min_events: int = 2
    coreg_blocks: int = 360
    layer_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "trajectory": 1.0,
            "emission_change": 1.0,
            "synchrony": 1.0,
            "validator_support": 1.0,
        }
    )
    layer_strong: float = 0.8
    min_layers: int = 2
    edge_min: float = 0.6
    resolution: float = 1.0
    min_size: int = 3
    min_density: float = 0.5
    null_draws: int = 200
    null_p: float = 0.05
    match_jaccard: float = 0.3
    min_track_windows: int = 3
    share_threshold: float = 0.15


@dataclass
class RewardConfig:
    """Settings for spotting reward spikes and classifying the mechanism behind them.

    Attributes
    ----------
    spike_baseline_days : float
        Length of the trailing baseline for each miner.
    spike_z : float
        A spike is a share at least ``spike_z`` standard deviations above baseline.
    spike_min_gain_factor : float
        A spike also needs a gain of at least ``factor / n_miners``.
    spike_sd_floor_factor : float
        Baseline standard deviation is floored at ``factor / n_miners``.
    support_ratio : float
        A spike is unsupported if its median-support gain is below ``ratio * emission gain``.
    repeat_min : int
        Number of unsupported spikes needed to count as a repeat offender.
    repeat_gap_days : float
        Spikes closer together than this count once.
    systematic_p : float
        Binomial p-value threshold for a systematic offender.
    churn_top_k : int
        Size of the top-k set used to measure turnover.
    mech_burn : float
        Median burn share at or above this classifies a subnet as burn_dominated.
    mech_wta_top1 : float
        Median top-1 miner share at or above this classifies it as winner_take_all.
    mech_concentrated_top10 : float
        Median top-10 share at or above this classifies it as concentrated.
    """

    spike_baseline_days: float = 3.0
    spike_z: float = 3.0
    spike_min_gain_factor: float = 1.0
    spike_sd_floor_factor: float = 0.1
    support_ratio: float = 0.5
    repeat_min: int = 2
    repeat_gap_days: float = 1.0
    systematic_p: float = 0.01
    churn_top_k: int = 10
    mech_burn: float = 0.5
    mech_wta_top1: float = 0.5
    mech_concentrated_top10: float = 0.7


@dataclass
class SufficiencyConfig:
    """Settings for testing whether the data is enough to predict emissions.

    Attributes
    ----------
    horizon_days : float
        How far ahead to predict emission or rank.
    coord_days : float
        Window for the future-coordination target.
    history_days : float
        Trailing window for volatility and synchrony features.
    train_frac : float
        Time split: the first part trains and the rest tests, with an embargo.
    max_rows_per_subnet : int
        Maximum number of rows sampled per subnet.
    max_train_rows : int
        Maximum number of rows used to fit.
    r2_insufficient : float
        R^2 below this means the signals are insufficient.
    r2_adequate : float
        R^2 at or above this means the signals are adequate.
    """

    horizon_days: float = 1.0
    coord_days: float = 2.0
    history_days: float = 2.0
    train_frac: float = 0.7
    max_rows_per_subnet: int = 15000
    max_train_rows: int = 300000
    r2_insufficient: float = 0.1
    r2_adequate: float = 0.3


@dataclass
class RankingConfig:
    """Bin edges that turn raw shares into 1-5 size and impact scores.

    Attributes
    ----------
    size_bands : list of float
        Share-of-subnets edges for size scores 1 through 5.
    impact_bands_pct : list of float
        Percent-of-emission edges for impact scores 1 through 5.
    """

    size_bands: List[float] = field(default_factory=lambda: [0.05, 0.15, 0.30, 0.50])
    impact_bands_pct: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 3.0])


@dataclass
class Config:
    """Top-level container for every settings group, plus a global seed.

    Attributes
    ----------
    paths : PathConfig
        File locations for inputs, cache, and results.
    general : GeneralConfig
        Settings shared across analyses.
    exploits : ExploitConfig
        Exploit detection settings.
    telemetry : TelemetryConfig
        Telemetry distinguishability settings.
    scoring : ScoringConfig
        Validator behavior and weight-quality settings.
    insights : InsightConfig
        Early-warning signal settings.
    sybil : SybilConfig
        Sybil cluster detection settings.
    rewards : RewardConfig
        Reward spike and mechanism settings.
    sufficiency : SufficiencyConfig
        Predictability test settings.
    ranking : RankingConfig
        Size and impact score bands.
    seed : int
        Random seed used by stochastic steps.
    """

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
        top_notes = _attribute_notes(Config)
        for sec in fields(self):
            obj = getattr(self, sec.name)
            if not hasattr(obj, "__dataclass_fields__"):
                rows.append(
                    {
                        "setting": sec.name,
                        "value": obj,
                        "note": top_notes.get(sec.name, ""),
                    }
                )
                continue
            notes = _attribute_notes(type(obj))
            for f in fields(obj):
                rows.append(
                    {
                        "setting": f"{sec.name}.{f.name}",
                        "value": getattr(obj, f.name),
                        "note": notes.get(f.name, ""),
                    }
                )
        return pd.DataFrame(rows)

    def search(self, term: str) -> pd.DataFrame:
        d = self.describe()
        t = term.lower()
        return d[
            d["setting"].str.lower().str.contains(t)
            | d["note"].str.lower().str.contains(t)
        ]


def _attribute_notes(cls) -> Dict[str, str]:
    """Read each attribute's description from the NumPy-style Attributes section."""
    doc = inspect.getdoc(cls) or ""
    lines = doc.splitlines()
    notes: Dict[str, str] = {}

    start = None
    for i in range(len(lines) - 1):
        if lines[i].strip() == "Attributes" and re.fullmatch(
            r"-+", lines[i + 1].strip()
        ):
            start = i + 2
            break
    if start is None:
        return notes

    name, parts = None, []
    for line in lines[start:]:
        if not line.strip():
            continue
        if not line.startswith(" "):
            if name:
                notes[name] = " ".join(parts)
            m = re.match(r"^(\w+)\s*:", line)
            name, parts = (m.group(1), []) if m else (None, [])
        elif name:
            parts.append(line.strip())
    if name:
        notes[name] = " ".join(parts)
    return notes
