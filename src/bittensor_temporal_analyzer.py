from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from numpy.typing import NDArray
from scipy.signal import periodogram
from scipy.stats import entropy, rankdata, spearmanr
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Optional example data structures
# ---------------------------------------------------------------------


@dataclass
class BlockInfo:
    block: int
    timestamp: Optional[Any] = None


@dataclass
class BlockSnapshot:
    block_info: Any
    netuid: List[int]

    # Arrays indexed first by subnet position in netuid.
    Active: Optional[List[NDArray]] = None
    ActivityCutoff: Optional[List[NDArray]] = None
    Bonds: Optional[List[NDArray]] = None
    Consensus: Optional[List[NDArray]] = None
    Dividends: Optional[List[NDArray]] = None
    Emission: Optional[List[NDArray]] = None
    Incentive: Optional[List[NDArray]] = None
    Kappa: Optional[List[NDArray]] = None
    LastUpdate: Optional[List[NDArray]] = None
    MechanismCountCurrent: Optional[List[NDArray]] = None
    NeuronCertificates: Optional[List[NDArray]] = None
    Rho: Optional[List[NDArray]] = None
    SubnetMechanism: Optional[List[NDArray]] = None
    Tempo: Optional[List[NDArray]] = None
    ValidatorPermit: Optional[List[NDArray]] = None
    ValidatorTrust: Optional[List[NDArray]] = None
    Weights: Optional[List[NDArray]] = None

    # Strongly recommended if available.
    # Each entry should contain stable identities for neurons in that subnet.
    Hotkeys: Optional[List[Sequence[str]]] = None


# ---------------------------------------------------------------------
# Complete single-class analysis pipeline
# ---------------------------------------------------------------------


class BittensorTemporalAnalyzer:
    """
    Complete temporal analysis pipeline for Bittensor block snapshots.

    Main outputs
    ------------
    1. miner_df
        Miner-level longitudinal table.

    2. subnet_df
        Subnet-level metrics per snapshot, including concentration,
        churn, weight instability, anomaly scores, and risk scores.

    3. mvs_df
        Miner-validator-subnet interaction metrics.

    4. telemetry_df
        Telemetry sufficiency and future-emission predictability results.

    5. signal_importance_df
        Permutation importance for predicting future emissions.

    6. sybil_pairs_df
        Miner pairs with similar, synchronized, validator-overlapping
        behavior.

    7. sybil_clusters_df
        Connected communities of Sybil-like miners.

    8. miner_risk_df
        Per-miner Sybil-like risk and behavior metrics.

    9. early_warning_df
        Evaluation of a temporal instability-warning model.

    Important assumptions
    ---------------------
    - Snapshot order is temporal after sorting by block/timestamp.
    - If Hotkeys are unavailable, neuron array positions are used as identities.
      This is weaker because UID positions may be recycled or reassigned.
    - Weight matrices are assumed to be validator-by-miner by default.
      Set weights_orientation="miner_by_validator" if necessary.
    - Sybil results are indicators, not proof of common ownership.
    - With only 100 snapshots, periodicity and ML results should be interpreted
      cautiously and validated against additional data.
    """

    MINER_SIGNALS = {
        "active": "Active",
        "activity_cutoff": "ActivityCutoff",
        "consensus": "Consensus",
        "dividends": "Dividends",
        "emission": "Emission",
        "incentive": "Incentive",
        "kappa": "Kappa",
        "last_update": "LastUpdate",
        "mechanism_count": "MechanismCountCurrent",
        "rho": "Rho",
        "subnet_mechanism": "SubnetMechanism",
        "tempo": "Tempo",
        "validator_permit": "ValidatorPermit",
        "validator_trust": "ValidatorTrust",
    }

    MODEL_FEATURES = [
        "active",
        "consensus",
        "dividends",
        "emission",
        "incentive",
        "kappa",
        "rho",
        "validator_permit",
        "validator_trust",
        "received_weight_sum",
        "received_weight_mean",
        "received_weight_std",
        "received_weight_max",
        "supporting_validator_count",
        "bond_received_sum",
        "emission_change",
        "consensus_change",
        "trust_change",
        "received_weight_change",
        "emission_rolling_mean",
        "emission_rolling_std",
        "consensus_rolling_mean",
        "trust_rolling_mean",
    ]

    SYBIL_TRAJECTORY_FEATURES = [
        "emission",
        "consensus",
        "incentive",
        "validator_trust",
        "received_weight_sum",
        "active",
    ]

    def __init__(
        self,
        snapshots: Sequence[Any],
        output_directory: str,
        weights_orientation: str = "validator_by_miner",
        minimum_history: int = 8,
        rolling_window: int = 5,
        future_horizon: int = 1,
        spike_z_threshold: float = 3.0,
        instability_quantile: float = 0.90,
        sybil_pair_threshold: float = 0.80,
        minimum_sybil_cluster_size: int = 2,
        anomaly_contamination: float = 0.10,
        random_state: int = 42,
    ) -> None:
        if not snapshots:
            raise ValueError("snapshots cannot be empty.")

        if weights_orientation not in {
            "validator_by_miner",
            "miner_by_validator",
            "auto",
        }:
            raise ValueError(
                "weights_orientation must be 'validator_by_miner', "
                "'miner_by_validator', or 'auto'."
            )

        self.snapshots = list(snapshots)
        self.output_directory = Path(output_directory)
        self.weights_orientation = weights_orientation
        self.minimum_history = minimum_history
        self.rolling_window = rolling_window
        self.future_horizon = future_horizon
        self.spike_z_threshold = spike_z_threshold
        self.instability_quantile = instability_quantile
        self.sybil_pair_threshold = sybil_pair_threshold
        self.minimum_sybil_cluster_size = minimum_sybil_cluster_size
        self.anomaly_contamination = anomaly_contamination
        self.random_state = random_state

        self.miner_df = pd.DataFrame()
        self.subnet_df = pd.DataFrame()
        self.mvs_df = pd.DataFrame()
        self.telemetry_df = pd.DataFrame()
        self.signal_importance_df = pd.DataFrame()
        self.sybil_pairs_df = pd.DataFrame()
        self.sybil_clusters_df = pd.DataFrame()
        self.miner_risk_df = pd.DataFrame()
        self.early_warning_df = pd.DataFrame()
        self.summary_df = pd.DataFrame()

        self.telemetry_model: Optional[Any] = None
        self.early_warning_model: Optional[Any] = None

    # =================================================================
    # General utility methods
    # =================================================================

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            array = np.asarray(value)

            if array.size == 0:
                return np.nan

            scalar = array.reshape(-1)[0]

            if scalar is None:
                return np.nan

            return float(scalar)
        except (TypeError, ValueError, IndexError):
            return np.nan

    @staticmethod
    def _safe_array(value: Any, dtype=float) -> np.ndarray:
        if value is None:
            return np.array([], dtype=dtype)

        try:
            array = np.asarray(value)

            if array.size == 0:
                return np.array([], dtype=dtype)

            return array.astype(dtype, copy=False)
        except (TypeError, ValueError):
            return np.array([], dtype=dtype)

    @staticmethod
    def _finite(values: Any) -> np.ndarray:
        values = np.asarray(values, dtype=float).ravel()
        return values[np.isfinite(values)]

    @staticmethod
    def _safe_mean(values: Any) -> float:
        values = BittensorTemporalAnalyzer._finite(values)
        return float(values.mean()) if values.size else np.nan

    @staticmethod
    def _safe_std(values: Any) -> float:
        values = BittensorTemporalAnalyzer._finite(values)
        return float(values.std(ddof=0)) if values.size else np.nan

    @staticmethod
    def _safe_corr(x: Any, y: Any) -> float:
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        size = min(x.size, y.size)
        if size < 3:
            return np.nan

        x = x[:size]
        y = y[:size]
        mask = np.isfinite(x) & np.isfinite(y)

        if mask.sum() < 3:
            return np.nan

        x = x[mask]
        y = y[mask]

        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan

        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def _gini(values: Any) -> float:
        x = BittensorTemporalAnalyzer._finite(values)

        if x.size == 0:
            return np.nan

        # Shift negative values if any, although most chain metrics should
        # be nonnegative.
        if np.min(x) < 0:
            x = x - np.min(x)

        total = np.sum(x)
        if total <= 0:
            return 0.0

        x = np.sort(x)
        n = x.size
        index = np.arange(1, n + 1)

        return float(np.sum((2 * index - n - 1) * x) / (n * total))

    @staticmethod
    def _normalized_entropy(values: Any) -> float:
        x = BittensorTemporalAnalyzer._finite(values)

        if x.size == 0:
            return np.nan

        x = np.clip(x, 0, None)
        total = x.sum()

        if total <= 0:
            return 0.0

        if x.size == 1:
            return 0.0

        probabilities = x / total
        return float(entropy(probabilities) / np.log(x.size))

    @staticmethod
    def _herfindahl(values: Any) -> float:
        x = BittensorTemporalAnalyzer._finite(values)

        if x.size == 0:
            return np.nan

        x = np.clip(x, 0, None)
        total = x.sum()

        if total <= 0:
            return 0.0

        shares = x / total
        return float(np.sum(shares**2))

    @staticmethod
    def _top_share(values: Any, fraction: float = 0.10) -> float:
        x = BittensorTemporalAnalyzer._finite(values)

        if x.size == 0:
            return np.nan

        x = np.clip(x, 0, None)
        total = x.sum()

        if total <= 0:
            return 0.0

        count = max(1, int(np.ceil(x.size * fraction)))
        return float(np.sort(x)[-count:].sum() / total)

    @staticmethod
    def _minmax_series(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        minimum = values.min()
        maximum = values.max()

        if not np.isfinite(minimum) or not np.isfinite(maximum):
            return pd.Series(np.nan, index=series.index)

        if np.isclose(maximum, minimum):
            return pd.Series(0.0, index=series.index)

        return (values - minimum) / (maximum - minimum)

    @staticmethod
    def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
        frame = pd.DataFrame({"x": x, "y": y}).dropna()

        if len(frame) < 3:
            return np.nan

        if frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
            return np.nan

        result = spearmanr(frame["x"], frame["y"])
        return float(result.statistic)

    @staticmethod
    def _extract_block(snapshot: Any, fallback: int) -> int:
        block_info = getattr(snapshot, "block_info", None)

        for attribute in ("block", "block_number", "number"):
            if block_info is not None and hasattr(block_info, attribute):
                try:
                    return int(getattr(block_info, attribute))
                except (TypeError, ValueError):
                    pass

        if isinstance(block_info, dict):
            for key in ("block", "block_number", "number"):
                if key in block_info:
                    try:
                        return int(block_info[key])
                    except (TypeError, ValueError):
                        pass

        return int(fallback)

    @staticmethod
    def _extract_timestamp(snapshot: Any) -> pd.Timestamp:
        block_info = getattr(snapshot, "block_info", None)

        value = None
        for attribute in ("timestamp", "datetime", "time", "date"):
            if block_info is not None and hasattr(block_info, attribute):
                value = getattr(block_info, attribute)
                break

        if value is None and isinstance(block_info, dict):
            for key in ("timestamp", "datetime", "time", "date"):
                if key in block_info:
                    value = block_info[key]
                    break

        if value is None:
            return pd.NaT

        try:
            return pd.to_datetime(value, utc=True)
        except Exception:
            return pd.NaT

    @staticmethod
    def _extract_subnet_value(
        snapshot: Any,
        field_name: str,
        subnet_position: int,
    ) -> Any:
        field = getattr(snapshot, field_name, None)

        if field is None:
            return None

        try:
            return field[subnet_position]
        except (IndexError, KeyError, TypeError):
            return None

    def _infer_neuron_count(
        self,
        snapshot: Any,
        subnet_position: int,
    ) -> int:
        candidate_fields = [
            "Emission",
            "Consensus",
            "Incentive",
            "Active",
            "ValidatorTrust",
            "ValidatorPermit",
            "LastUpdate",
            "Hotkeys",
        ]
        counts = []
        for field_name in candidate_fields:
            value = self._extract_subnet_value(snapshot, field_name, subnet_position)
            if value is None:
                continue
            try:
                array = np.asarray(value)
                if array.ndim >= 1:
                    counts.append(int(array.shape[0]))
            except Exception:
                continue
        return max(counts) if counts else 0

    def _get_identity(
        self,
        snapshot: Any,
        subnet_position: int,
        neuron_index: int,
        netuid: int,
    ) -> str:
        hotkeys = self._extract_subnet_value(snapshot, "Hotkeys", subnet_position)

        if hotkeys is not None:
            try:
                hotkey = str(hotkeys[neuron_index])
                if hotkey and hotkey.lower() != "nan":
                    return hotkey
            except (IndexError, TypeError):
                pass

        return f"netuid_{netuid}_uid_{neuron_index}"

    def _get_neuron_value(
        self,
        snapshot: Any,
        field_name: str,
        subnet_position: int,
        neuron_index: int,
    ) -> float:
        value = self._extract_subnet_value(snapshot, field_name, subnet_position)

        if value is None:
            return np.nan

        try:
            array = np.asarray(value)
            if array.ndim == 0:
                return self._safe_float(array)
            if neuron_index >= array.shape:
                return np.nan
            return self._safe_float(array[neuron_index])
        except Exception:
            return np.nan

    def _orient_matrix(
        self,
        matrix: Any,
        neuron_count: Optional[int] = None,
    ) -> np.ndarray:
        array = self._safe_array(matrix, dtype=float)

        if array.size == 0:
            return np.empty((0, 0), dtype=float)

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            return np.empty((0, 0), dtype=float)

        if self.weights_orientation == "validator_by_miner":
            return array

        if self.weights_orientation == "miner_by_validator":
            return array.T

        # Auto mode. Prefer an axis matching the known neuron count as
        # the miner dimension.
        if neuron_count is not None:
            if array.shape[1] == neuron_count:
                return array
            if array.shape[0] == neuron_count:
                return array.T

        # For square matrices, Bittensor-like weight matrices are usually
        # interpreted row -> column, or validator -> miner.
        return array

    @staticmethod
    def _matrix_distance(
        current: np.ndarray,
        previous: Optional[np.ndarray],
    ) -> float:
        if previous is None or current.size == 0 or previous.size == 0:
            return np.nan

        rows = min(current.shape[0], previous.shape[0])
        columns = min(current.shape[1], previous.shape[1])

        if rows == 0 or columns == 0:
            return np.nan

        difference = current[:rows, :columns] - previous[:rows, :columns]

        denominator = np.sqrt(rows * columns)
        return float(np.linalg.norm(difference, ord="fro") / denominator)

    # =================================================================
    # Data preparation
    # =================================================================

    def build_miner_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        sorted_snapshots = sorted(
            enumerate(self.snapshots),
            key=lambda item: self._extract_block(item[1], item[0]),
        )

        for temporal_position, (_, snapshot) in enumerate(sorted_snapshots):
            block = self._extract_block(snapshot, temporal_position)
            timestamp = self._extract_timestamp(snapshot)
            netuids = list(getattr(snapshot, "netuid", []))

            for subnet_position, raw_netuid in enumerate(netuids):
                netuid = int(raw_netuid)
                neuron_count = self._infer_neuron_count(snapshot, subnet_position)

                weights_raw = self._extract_subnet_value(
                    snapshot, "Weights", subnet_position
                )
                bonds_raw = self._extract_subnet_value(
                    snapshot, "Bonds", subnet_position
                )

                weights = self._orient_matrix(weights_raw, neuron_count=neuron_count)
                bonds = self._orient_matrix(bonds_raw, neuron_count=neuron_count)

                for neuron_index in range(neuron_count):
                    miner_id = self._get_identity(
                        snapshot,
                        subnet_position,
                        neuron_index,
                        netuid,
                    )

                    row: Dict[str, Any] = {
                        "snapshot_index": temporal_position,
                        "block": block,
                        "timestamp": timestamp,
                        "netuid": netuid,
                        "subnet_position": subnet_position,
                        "miner_index": neuron_index,
                        "miner_id": miner_id,
                    }

                    for output_name, field_name in self.MINER_SIGNALS.items():
                        row[output_name] = self._get_neuron_value(
                            snapshot,
                            field_name,
                            subnet_position,
                            neuron_index,
                        )

                    # Incoming validator support for this miner.
                    if weights.size > 0 and neuron_index < weights.shape[1]:
                        received = weights[:, neuron_index]
                        finite_received = received[np.isfinite(received)]

                        row["received_weight_sum"] = (
                            float(np.sum(finite_received))
                            if finite_received.size
                            else np.nan
                        )
                        row["received_weight_mean"] = (
                            float(np.mean(finite_received))
                            if finite_received.size
                            else np.nan
                        )
                        row["received_weight_std"] = (
                            float(np.std(finite_received))
                            if finite_received.size
                            else np.nan
                        )
                        row["received_weight_max"] = (
                            float(np.max(finite_received))
                            if finite_received.size
                            else np.nan
                        )
                        row["supporting_validator_count"] = (
                            int(np.sum(finite_received > 0))
                            if finite_received.size
                            else 0
                        )
                    else:
                        row["received_weight_sum"] = np.nan
                        row["received_weight_mean"] = np.nan
                        row["received_weight_std"] = np.nan
                        row["received_weight_max"] = np.nan
                        row["supporting_validator_count"] = np.nan

                    if bonds.size > 0 and neuron_index < bonds.shape:
                        received_bonds = bonds[:, neuron_index]
                        row["bond_received_sum"] = self._safe_mean(
                            [np.nansum(received_bonds)]
                        )
                    else:
                        row["bond_received_sum"] = np.nan

                    rows.append(row)

        dataframe = pd.DataFrame(rows)

        if dataframe.empty:
            raise ValueError("No miner records could be extracted from the snapshots.")

        dataframe = dataframe.sort_values(
            ["netuid", "miner_id", "block", "snapshot_index"]
        ).reset_index(drop=True)

        group_columns = ["netuid", "miner_id"]
        grouped = dataframe.groupby(group_columns, sort=False)

        change_columns = {
            "emission": "emission_change",
            "consensus": "consensus_change",
            "validator_trust": "trust_change",
            "received_weight_sum": "received_weight_change",
            "incentive": "incentive_change",
        }

        for source, destination in change_columns.items():
            dataframe[destination] = grouped[source].diff()

        rolling_columns = [
            "emission",
            "consensus",
            "validator_trust",
            "received_weight_sum",
        ]

        for column in rolling_columns:
            dataframe[f"{column}_rolling_mean"] = grouped[column].transform(
                lambda series: series.rolling(
                    self.rolling_window,
                    min_periods=2,
                ).mean()
            )

            dataframe[f"{column}_rolling_std"] = grouped[column].transform(
                lambda series: series.rolling(
                    self.rolling_window,
                    min_periods=2,
                ).std(ddof=0)
            )

        emission_grouped = dataframe.groupby(group_columns, sort=False)["emission"]

        expanding_mean = emission_grouped.transform(
            lambda series: series.shift(1)
            .expanding(min_periods=max(3, self.rolling_window))
            .mean()
        )
        expanding_std = emission_grouped.transform(
            lambda series: series.shift(1)
            .expanding(min_periods=max(3, self.rolling_window))
            .std(ddof=0)
        )

        dataframe["emission_z_score"] = (
            dataframe["emission"] - expanding_mean
        ) / expanding_std.replace(0, np.nan)

        dataframe["emission_spike"] = (
            dataframe["emission_z_score"] >= self.spike_z_threshold
        ).astype(int)

        dataframe["future_emission"] = grouped["emission"].shift(-self.future_horizon)
        dataframe["future_consensus"] = grouped["consensus"].shift(-self.future_horizon)

        dataframe["future_emission_change"] = (
            dataframe["future_emission"] - dataframe["emission"]
        )

        # Rank within a subnet snapshot.
        dataframe["emission_rank"] = dataframe.groupby(
            ["netuid", "block", "snapshot_index"],
            sort=False,
        )["emission"].rank(
            ascending=False,
            method="average",
        )

        dataframe["previous_emission_rank"] = grouped["emission_rank"].shift(1)

        dataframe["absolute_rank_change"] = (
            dataframe["emission_rank"] - dataframe["previous_emission_rank"]
        ).abs()

        self.miner_df = dataframe
        return self.miner_df

    # =================================================================
    # Subnet and miner-behavior analysis
    # =================================================================

    def build_subnet_metrics(self) -> pd.DataFrame:
        if self.miner_df.empty:
            self.build_miner_dataframe()

        rows: List[Dict[str, Any]] = []
        previous_weights: Dict[int, np.ndarray] = {}
        previous_emissions: Dict[int, pd.Series] = {}

        sorted_snapshots = sorted(
            enumerate(self.snapshots),
            key=lambda item: self._extract_block(item[1], item[0]),
        )

        for temporal_position, (_, snapshot) in enumerate(sorted_snapshots):
            block = self._extract_block(snapshot, temporal_position)
            timestamp = self._extract_timestamp(snapshot)
            netuids = list(getattr(snapshot, "netuid", []))

            for subnet_position, raw_netuid in enumerate(netuids):
                netuid = int(raw_netuid)

                group = self.miner_df[
                    (self.miner_df["netuid"] == netuid)
                    & (self.miner_df["block"] == block)
                    & (self.miner_df["snapshot_index"] == temporal_position)
                ].copy()

                if group.empty:
                    continue

                neuron_count = len(group)
                weights_raw = self._extract_subnet_value(
                    snapshot, "Weights", subnet_position
                )
                weights = self._orient_matrix(weights_raw, neuron_count=neuron_count)

                emissions = group["emission"].to_numpy(dtype=float)
                consensus = group["consensus"].to_numpy(dtype=float)
                trust = group["validator_trust"].to_numpy(dtype=float)
                incentives = group["incentive"].to_numpy(dtype=float)

                current_emissions = group.set_index("miner_id")["emission"]
                previous = previous_emissions.get(netuid)

                if previous is not None:
                    aligned = pd.concat(
                        [
                            previous.rename("previous"),
                            current_emissions.rename("current"),
                        ],
                        axis=1,
                    ).fillna(0.0)

                    previous_ranks = rankdata(-aligned["previous"].to_numpy())
                    current_ranks = rankdata(-aligned["current"].to_numpy())

                    rank_churn = float(np.mean(np.abs(current_ranks - previous_ranks)))
                    participant_churn = float(
                        len(
                            set(previous.index).symmetric_difference(
                                set(current_emissions.index)
                            )
                        )
                        / max(
                            1,
                            len(
                                set(previous.index).union(set(current_emissions.index))
                            ),
                        )
                    )
                else:
                    rank_churn = np.nan
                    participant_churn = np.nan

                previous_emissions[netuid] = current_emissions

                if weights.size:
                    validator_totals = np.nansum(np.clip(weights, 0, None), axis=1)
                    miner_totals = np.nansum(np.clip(weights, 0, None), axis=0)
                    positive_edges = int(np.sum(weights > 0))
                    possible_edges = int(weights.size)
                    graph_density = (
                        positive_edges / possible_edges if possible_edges else np.nan
                    )
                else:
                    validator_totals = np.array([])
                    miner_totals = np.array([])
                    graph_density = np.nan

                weight_volatility = self._matrix_distance(
                    weights,
                    previous_weights.get(netuid),
                )

                if weights.size:
                    previous_weights[netuid] = weights.copy()

                emission_consensus_corr = self._safe_corr(emissions, consensus)
                emission_trust_corr = self._safe_corr(emissions, trust)

                row = {
                    "snapshot_index": temporal_position,
                    "block": block,
                    "timestamp": timestamp,
                    "netuid": netuid,
                    "miner_count": int(len(group)),
                    "active_miner_count": int(
                        np.nansum(group["active"].fillna(0).to_numpy() > 0)
                    ),
                    "validator_count": (
                        int(weights.shape[0]) if weights.size else np.nan
                    ),
                    "emission_total": float(np.nansum(emissions)),
                    "emission_mean": self._safe_mean(emissions),
                    "emission_std": self._safe_std(emissions),
                    "emission_gini": self._gini(emissions),
                    "emission_entropy": self._normalized_entropy(emissions),
                    "emission_hhi": self._herfindahl(emissions),
                    "emission_top_10pct_share": self._top_share(
                        emissions, fraction=0.10
                    ),
                    "consensus_mean": self._safe_mean(consensus),
                    "consensus_std": self._safe_std(consensus),
                    "trust_mean": self._safe_mean(trust),
                    "trust_std": self._safe_std(trust),
                    "incentive_mean": self._safe_mean(incentives),
                    "validator_concentration": self._gini(validator_totals),
                    "miner_weight_concentration": self._gini(miner_totals),
                    "weight_graph_density": graph_density,
                    "weight_volatility": weight_volatility,
                    "rank_churn": rank_churn,
                    "participant_churn": participant_churn,
                    "spike_rate": float(group["emission_spike"].mean()),
                    "mean_absolute_rank_change": self._safe_mean(
                        group["absolute_rank_change"]
                    ),
                    "emission_consensus_corr": emission_consensus_corr,
                    "emission_trust_corr": emission_trust_corr,
                    "emission_consensus_divergence": (
                        1.0 - emission_consensus_corr
                        if np.isfinite(emission_consensus_corr)
                        else np.nan
                    ),
                    "mechanism_count": self._safe_mean(group["mechanism_count"]),
                    "subnet_mechanism": self._safe_mean(group["subnet_mechanism"]),
                    "tempo": self._safe_mean(group["tempo"]),
                }

                rows.append(row)

        dataframe = (
            pd.DataFrame(rows)
            .sort_values(["netuid", "block", "snapshot_index"])
            .reset_index(drop=True)
        )

        grouped = dataframe.groupby("netuid", sort=False)

        mechanism_columns = [
            "mechanism_count",
            "subnet_mechanism",
            "tempo",
        ]

        mechanism_change = pd.Series(False, index=dataframe.index, dtype=bool)

        for column in mechanism_columns:
            changed = grouped[column].transform(
                lambda series: (
                    series.notna()
                    & series.shift(1).notna()
                    & series.ne(series.shift(1))
                )
            )
            mechanism_change = mechanism_change | changed

        dataframe["mechanism_change"] = mechanism_change.astype(int)

        instability_components = [
            "emission_gini",
            "emission_hhi",
            "weight_volatility",
            "rank_churn",
            "participant_churn",
            "spike_rate",
            "emission_consensus_divergence",
        ]

        for column in instability_components:
            dataframe[f"{column}_scaled"] = dataframe.groupby("netuid", sort=False)[
                column
            ].transform(self._minmax_series)

        dataframe["raw_instability_score"] = (
            0.20 * dataframe["emission_gini_scaled"].fillna(0)
            + 0.10 * dataframe["emission_hhi_scaled"].fillna(0)
            + 0.20 * dataframe["weight_volatility_scaled"].fillna(0)
            + 0.15 * dataframe["rank_churn_scaled"].fillna(0)
            + 0.10 * dataframe["participant_churn_scaled"].fillna(0)
            + 0.10 * dataframe["spike_rate_scaled"].fillna(0)
            + 0.15 * dataframe["emission_consensus_divergence_scaled"].fillna(0)
        )

        # Future instability allows earlier snapshots to be used as features.
        dataframe["future_instability_score"] = grouped["raw_instability_score"].shift(
            -self.future_horizon
        )

        threshold = dataframe["future_instability_score"].quantile(
            self.instability_quantile
        )

        dataframe["future_instability_event"] = (
            dataframe["future_instability_score"] >= threshold
        ).astype("Int64")

        self.subnet_df = dataframe
        return self.subnet_df

    # =================================================================
    # MVS analysis
    # =================================================================

    def analyze_mvs_interactions(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        sorted_snapshots = sorted(
            enumerate(self.snapshots),
            key=lambda item: self._extract_block(item[1], item[0]),
        )

        for temporal_position, (_, snapshot) in enumerate(sorted_snapshots):
            block = self._extract_block(snapshot, temporal_position)
            timestamp = self._extract_timestamp(snapshot)
            netuids = list(getattr(snapshot, "netuid", []))

            for subnet_position, raw_netuid in enumerate(netuids):
                netuid = int(raw_netuid)
                neuron_count = self._infer_neuron_count(snapshot, subnet_position)

                weights = self._orient_matrix(
                    self._extract_subnet_value(snapshot, "Weights", subnet_position),
                    neuron_count=neuron_count,
                )

                if weights.size == 0:
                    continue

                graph = nx.DiGraph()

                for validator_index in range(weights.shape[0]):
                    validator = f"netuid_{netuid}_validator_{validator_index}"
                    graph.add_node(
                        validator,
                        node_type="validator",
                    )

                for miner_index in range(weights.shape[1]):
                    miner = f"netuid_{netuid}_miner_{miner_index}"
                    graph.add_node(miner, node_type="miner")

                for validator_index in range(weights.shape[0]):
                    for miner_index in range(weights.shape[1]):
                        weight = weights[validator_index, miner_index]

                        if np.isfinite(weight) and weight > 0:
                            graph.add_edge(
                                f"netuid_{netuid}_validator_{validator_index}",
                                f"netuid_{netuid}_miner_{miner_index}",
                                weight=float(weight),
                            )

                validator_strengths = {
                    node: float(
                        sum(
                            data.get("weight", 0.0)
                            for _, _, data in graph.out_edges(node, data=True)
                        )
                    )
                    for node, data in graph.nodes(data=True)
                    if data["node_type"] == "validator"
                }

                miner_strengths = {
                    node: float(
                        sum(
                            data.get("weight", 0.0)
                            for _, _, data in graph.in_edges(node, data=True)
                        )
                    )
                    for node, data in graph.nodes(data=True)
                    if data["node_type"] == "miner"
                }

                degree_centrality = nx.degree_centrality(graph)

                if graph.number_of_nodes() > 1:
                    try:
                        pagerank = nx.pagerank(
                            graph,
                            weight="weight",
                            max_iter=500,
                        )
                    except Exception:
                        pagerank = {node: np.nan for node in graph.nodes}
                else:
                    pagerank = {node: np.nan for node in graph.nodes}

                row = {
                    "snapshot_index": temporal_position,
                    "block": block,
                    "timestamp": timestamp,
                    "netuid": netuid,
                    "validator_count": int(weights.shape[0]),
                    "miner_count": int(weights.shape[1]),
                    "edge_count": int(graph.number_of_edges()),
                    "graph_density": float(nx.density(graph)),
                    "validator_weight_gini": self._gini(
                        list(validator_strengths.values())
                    ),
                    "miner_received_weight_gini": self._gini(
                        list(miner_strengths.values())
                    ),
                    "largest_validator_weight_share": (
                        max(validator_strengths.values())
                        / max(
                            1e-12,
                            sum(validator_strengths.values()),
                        )
                        if validator_strengths
                        else np.nan
                    ),
                    "largest_miner_weight_share": (
                        max(miner_strengths.values())
                        / max(
                            1e-12,
                            sum(miner_strengths.values()),
                        )
                        if miner_strengths
                        else np.nan
                    ),
                    "mean_validator_degree_centrality": self._safe_mean(
                        [degree_centrality[node] for node in validator_strengths]
                    ),
                    "mean_miner_pagerank": self._safe_mean(
                        [pagerank[node] for node in miner_strengths]
                    ),
                }

                rows.append(row)

        self.mvs_df = pd.DataFrame(rows)
        return self.mvs_df

    # =================================================================
    # Telemetry sufficiency and signal importance
    # =================================================================

    def evaluate_telemetry_sufficiency(self) -> pd.DataFrame:
        if self.miner_df.empty:
            self.build_miner_dataframe()

        available_features = [
            feature
            for feature in self.MODEL_FEATURES
            if feature in self.miner_df.columns and self.miner_df[feature].notna().any()
        ]

        model_data = self.miner_df[
            available_features + ["future_emission", "netuid", "block", "miner_id"]
        ].copy()

        model_data = model_data.dropna(subset=["future_emission"])

        if len(model_data) < 20:
            warnings.warn("Too few observations for telemetry modeling.")
            self.telemetry_df = pd.DataFrame()
            self.signal_importance_df = pd.DataFrame()
            return self.telemetry_df

        unique_blocks = np.sort(model_data["block"].unique())

        if len(unique_blocks) < 4:
            warnings.warn("At least four distinct blocks are recommended.")
            self.telemetry_df = pd.DataFrame()
            self.signal_importance_df = pd.DataFrame()
            return self.telemetry_df

        split_position = max(1, int(np.floor(len(unique_blocks) * 0.80)))
        training_blocks = unique_blocks[:split_position]
        testing_blocks = unique_blocks[split_position:]

        if len(testing_blocks) == 0:
            testing_blocks = unique_blocks[-1:]
            training_blocks = unique_blocks[:-1]

        training = model_data[model_data["block"].isin(training_blocks)]
        testing = model_data[model_data["block"].isin(testing_blocks)]

        X_train = training[available_features]
        y_train = training["future_emission"]
        X_test = testing[available_features]
        y_test = testing["future_emission"]

        model = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        baseline_prediction = np.repeat(y_train.median(), len(y_test))
        baseline_mae = mean_absolute_error(y_test, baseline_prediction)

        normalized_mae = mae / (np.mean(np.abs(y_test)) + 1e-12)
        improvement_over_baseline = 1.0 - (mae / (baseline_mae + 1e-12))

        # Lower predictability implies greater telemetry insufficiency.
        bounded_r2 = float(np.clip(r2, 0.0, 1.0))
        telemetry_insufficiency_score = float(
            np.clip(
                0.60 * (1.0 - bounded_r2) + 0.40 * np.clip(normalized_mae, 0.0, 1.0),
                0.0,
                1.0,
            )
        )

        self.telemetry_model = model

        self.telemetry_df = pd.DataFrame(
            [
                {
                    "training_observations": len(training),
                    "testing_observations": len(testing),
                    "feature_count": len(available_features),
                    "training_start_block": min(training_blocks),
                    "training_end_block": max(training_blocks),
                    "testing_start_block": min(testing_blocks),
                    "testing_end_block": max(testing_blocks),
                    "r2": r2,
                    "mae": mae,
                    "rmse": rmse,
                    "normalized_mae": normalized_mae,
                    "baseline_mae": baseline_mae,
                    "improvement_over_baseline": (improvement_over_baseline),
                    "telemetry_insufficiency_score": (telemetry_insufficiency_score),
                }
            ]
        )

        importance = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.signal_importance_df = (
            pd.DataFrame(
                {
                    "feature": available_features,
                    "importance_mean": importance.importances_mean,
                    "importance_std": importance.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

        total_positive_importance = (
            self.signal_importance_df["importance_mean"].clip(lower=0).sum()
        )

        self.signal_importance_df["importance_share"] = self.signal_importance_df[
            "importance_mean"
        ].clip(lower=0) / (total_positive_importance + 1e-12)

        return self.telemetry_df

    # =================================================================
    # Periodicity and miner adaptation
    # =================================================================

    def _estimate_cycle(
        self,
        series: pd.Series,
        timestamps: pd.Series,
    ) -> Tuple[float, float]:
        values = pd.to_numeric(series, errors="coerce").to_numpy()
        mask = np.isfinite(values)
        values = values[mask]

        if values.size < max(8, self.minimum_history):
            return np.nan, np.nan

        values = values - np.mean(values)

        if np.std(values) == 0:
            return np.nan, 0.0

        frequencies, power = periodogram(values)

        valid = frequencies > 0
        frequencies = frequencies[valid]
        power = power[valid]

        if frequencies.size == 0 or np.sum(power) <= 0:
            return np.nan, np.nan

        peak_index = int(np.argmax(power))
        cycle_in_snapshots = 1.0 / frequencies[peak_index]
        cycle_strength = float(power[peak_index] / np.sum(power))

        timestamp_values = pd.to_datetime(timestamps, errors="coerce", utc=True)
        timestamp_values = timestamp_values[timestamp_values.notna()]

        if len(timestamp_values) >= 2:
            differences = (
                timestamp_values.sort_values().diff().dropna().dt.total_seconds()
                / 86400.0
            )
            median_days = differences.median()
            cycle_days = (
                cycle_in_snapshots * median_days if np.isfinite(median_days) else np.nan
            )
        else:
            cycle_days = np.nan

        return float(cycle_days), cycle_strength

    def build_miner_behavior_metrics(self) -> pd.DataFrame:
        if self.miner_df.empty:
            self.build_miner_dataframe()

        rows = []

        for (netuid, miner_id), group in self.miner_df.groupby(
            ["netuid", "miner_id"], sort=False
        ):
            group = group.sort_values(["block", "snapshot_index"])

            if len(group) < self.minimum_history:
                continue

            cycle_days, cycle_strength = self._estimate_cycle(
                group["emission"],
                group["timestamp"],
            )

            emission_consensus_corr = self._safe_corr(
                group["emission"], group["consensus"]
            )
            emission_trust_corr = self._safe_corr(
                group["emission"], group["validator_trust"]
            )
            emission_weight_corr = self._safe_corr(
                group["emission"], group["received_weight_sum"]
            )

            divergence = (
                1.0 - emission_consensus_corr
                if np.isfinite(emission_consensus_corr)
                else np.nan
            )

            rows.append(
                {
                    "netuid": int(netuid),
                    "miner_id": miner_id,
                    "observation_count": len(group),
                    "mean_emission": group["emission"].mean(),
                    "emission_volatility": group["emission"].std(ddof=0),
                    "emission_change_volatility": group["emission_change"].std(ddof=0),
                    "spike_rate": group["emission_spike"].mean(),
                    "mean_rank_change": group["absolute_rank_change"].mean(),
                    "emission_consensus_corr": (emission_consensus_corr),
                    "emission_trust_corr": emission_trust_corr,
                    "emission_weight_corr": emission_weight_corr,
                    "emission_consensus_divergence": divergence,
                    "candidate_cycle_days": cycle_days,
                    "candidate_cycle_strength": cycle_strength,
                    "candidate_2_to_4_week_cycle": (
                        int(
                            np.isfinite(cycle_days)
                            and 14 <= cycle_days <= 28
                            and cycle_strength >= 0.20
                        )
                    ),
                }
            )

        return pd.DataFrame(rows)

    # =================================================================
    # Sybil-candidate analysis
    # =================================================================

    @staticmethod
    def _synchronized_update_score(
        group_a: pd.DataFrame,
        group_b: pd.DataFrame,
    ) -> float:
        merged = group_a[["block", "last_update"]].merge(
            group_b[["block", "last_update"]],
            on="block",
            suffixes=("_a", "_b"),
        )

        if len(merged) < 3:
            return np.nan

        valid = merged.dropna()
        if len(valid) < 3:
            return np.nan

        differences = (valid["last_update_a"] - valid["last_update_b"]).abs()

        # Exact or near-exact update blocks are considered synchronized.
        return float(np.mean(differences <= 1))

    @staticmethod
    def _validator_overlap(
        vector_a: np.ndarray,
        vector_b: np.ndarray,
    ) -> float:
        size = min(vector_a.size, vector_b.size)

        if size == 0:
            return np.nan

        support_a = set(np.where(vector_a[:size] > 0)[0])
        support_b = set(np.where(vector_b[:size] > 0)[0])

        union = support_a | support_b
        if not union:
            return 0.0

        return float(len(support_a & support_b) / len(union))

    def _mean_received_weight_vector(
        self,
        netuid: int,
        miner_id: str,
    ) -> np.ndarray:
        vectors = []

        sorted_snapshots = sorted(
            enumerate(self.snapshots),
            key=lambda item: self._extract_block(item[1], item[0]),
        )

        for temporal_position, (_, snapshot) in enumerate(sorted_snapshots):
            netuids = list(getattr(snapshot, "netuid", []))

            if netuid not in netuids:
                continue

            subnet_position = netuids.index(netuid)
            neuron_count = self._infer_neuron_count(snapshot, subnet_position)

            identities = [
                self._get_identity(
                    snapshot,
                    subnet_position,
                    index,
                    netuid,
                )
                for index in range(neuron_count)
            ]

            if miner_id not in identities:
                continue

            miner_index = identities.index(miner_id)
            weights = self._orient_matrix(
                self._extract_subnet_value(snapshot, "Weights", subnet_position),
                neuron_count=neuron_count,
            )

            if weights.size > 0 and miner_index < weights.shape[1]:
                vectors.append(weights[:, miner_index])

        if not vectors:
            return np.array([])

        max_length = max(vector.size for vector in vectors)
        padded = np.full((len(vectors), max_length), np.nan)

        for row_index, vector in enumerate(vectors):
            padded[row_index, : vector.size] = vector

        return np.nanmean(padded, axis=0)

    def detect_sybil_candidates(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect miner pairs and communities exhibiting Sybil-like behavior.

        The method evaluates miners within the same subnet using:

        1. Temporal trajectory similarity across chain signals.
        2. Synchronization of LastUpdate values.
        3. Overlap in validator support.
        4. Emission trajectory similarity.
        5. Graph-based clustering of high-risk miner pairs.

        Returns
        -------
        sybil_pairs_df : pd.DataFrame
            One row per evaluated miner pair.

        sybil_clusters_df : pd.DataFrame
            One row per detected Sybil-like miner cluster.

        Notes
        -----
        - These results indicate behavioral similarity, not proof that miners
        are controlled by the same entity.
        - Stable miner identities such as hotkeys should be used whenever
        possible.
        """

        if self.miner_df.empty:
            self.build_miner_dataframe()

        pair_rows: List[Dict[str, Any]] = []
        cluster_rows: List[Dict[str, Any]] = []
        miner_risk_rows: List[Dict[str, Any]] = []

        # ---------------------------------------------------------------
        # Analyze each subnet independently
        # ---------------------------------------------------------------

        for netuid, subnet_group in self.miner_df.groupby(
            "netuid",
            sort=False,
        ):
            subnet_group = subnet_group.sort_values(["block", "snapshot_index"])

            # Keep only miners with enough temporal observations.
            miner_groups = {
                miner_id: group.sort_values(["block", "snapshot_index"]).copy()
                for miner_id, group in subnet_group.groupby(
                    "miner_id",
                    sort=False,
                )
                if len(group) >= self.minimum_history
            }

            miner_ids = sorted(miner_groups.keys())

            if len(miner_ids) < 2:
                continue

            # -----------------------------------------------------------
            # Build average validator-support vectors for every miner
            # -----------------------------------------------------------

            validator_vectors = {
                miner_id: self._mean_received_weight_vector(
                    int(netuid),
                    miner_id,
                )
                for miner_id in miner_ids
            }

            # Candidate graph:
            # node = miner
            # edge = miner pair above Sybil threshold
            candidate_graph = nx.Graph()
            candidate_graph.add_nodes_from(miner_ids)

            miner_pair_scores: Dict[str, List[float]] = {
                miner_id: [] for miner_id in miner_ids
            }

            candidate_neighbor_scores: Dict[str, List[float]] = {
                miner_id: [] for miner_id in miner_ids
            }

            # -----------------------------------------------------------
            # Compare every miner pair
            # -----------------------------------------------------------

            for first_index in range(len(miner_ids)):
                for second_index in range(
                    first_index + 1,
                    len(miner_ids),
                ):
                    miner_a = miner_ids[first_index]
                    miner_b = miner_ids[second_index]

                    group_a = miner_groups[miner_a]
                    group_b = miner_groups[miner_b]

                    available_features = [
                        feature
                        for feature in self.SYBIL_TRAJECTORY_FEATURES
                        if feature in group_a.columns and feature in group_b.columns
                    ]

                    if not available_features:
                        continue

                    # Align both miners using snapshots, not only block number.
                    merge_keys = ["block"]

                    if (
                        "snapshot_index" in group_a.columns
                        and "snapshot_index" in group_b.columns
                    ):
                        merge_keys.append("snapshot_index")

                    merged = group_a[merge_keys + available_features].merge(
                        group_b[merge_keys + available_features],
                        on=merge_keys,
                        suffixes=("_a", "_b"),
                        how="inner",
                    )

                    if len(merged) < self.minimum_history:
                        continue

                    # ---------------------------------------------------
                    # 1. Per-signal trajectory similarity
                    # ---------------------------------------------------

                    feature_similarity_values: Dict[str, float] = {}

                    for feature in available_features:
                        values_a = pd.to_numeric(
                            merged[f"{feature}_a"],
                            errors="coerce",
                        ).to_numpy(dtype=float)

                        values_b = pd.to_numeric(
                            merged[f"{feature}_b"],
                            errors="coerce",
                        ).to_numpy(dtype=float)

                        finite_mask = np.isfinite(values_a) & np.isfinite(values_b)

                        if finite_mask.sum() < 3:
                            feature_similarity_values[feature] = np.nan
                            continue

                        values_a = values_a[finite_mask]
                        values_b = values_b[finite_mask]

                        std_a = np.std(values_a)
                        std_b = np.std(values_b)

                        if np.isclose(std_a, 0) and np.isclose(std_b, 0):
                            # Both trajectories are constant.
                            scale = max(
                                abs(np.mean(values_a)),
                                abs(np.mean(values_b)),
                                1e-12,
                            )

                            normalized_difference = (
                                abs(np.mean(values_a) - np.mean(values_b)) / scale
                            )

                            similarity = float(
                                np.clip(
                                    1.0 - normalized_difference,
                                    0.0,
                                    1.0,
                                )
                            )

                        elif np.isclose(std_a, 0) or np.isclose(std_b, 0):
                            # One miner varies while the other does not.
                            similarity = 0.0

                        else:
                            correlation = np.corrcoef(
                                values_a,
                                values_b,
                            )[0, 1]

                            # Convert correlation from [-1, 1] to [0, 1].
                            similarity = float(
                                np.clip(
                                    (correlation + 1.0) / 2.0,
                                    0.0,
                                    1.0,
                                )
                            )

                        feature_similarity_values[feature] = similarity

                    valid_feature_similarities = [
                        value
                        for value in feature_similarity_values.values()
                        if np.isfinite(value)
                    ]

                    trajectory_similarity = (
                        float(np.mean(valid_feature_similarities))
                        if valid_feature_similarities
                        else np.nan
                    )

                    # ---------------------------------------------------
                    # 2. Emission-specific similarity
                    # ---------------------------------------------------

                    emission_similarity = np.nan
                    emission_correlation = np.nan
                    emission_change_similarity = np.nan

                    if "emission_a" in merged.columns and "emission_b" in merged.columns:
                        emission_frame = (
                            merged[["emission_a", "emission_b"]]
                            .apply(
                                pd.to_numeric,
                                errors="coerce",
                            )
                            .dropna()
                        )

                        if len(emission_frame) >= 3:
                            emission_a = emission_frame["emission_a"].to_numpy(dtype=float)

                            emission_b = emission_frame["emission_b"].to_numpy(dtype=float)

                            std_a = np.std(emission_a)
                            std_b = np.std(emission_b)

                            if not np.isclose(std_a, 0) and not np.isclose(std_b, 0):
                                emission_correlation = float(
                                    np.corrcoef(
                                        emission_a,
                                        emission_b,
                                    )[0, 1]
                                )

                                emission_similarity = float(
                                    np.clip(
                                        (emission_correlation + 1.0) / 2.0,
                                        0.0,
                                        1.0,
                                    )
                                )

                            elif np.isclose(std_a, 0) and np.isclose(std_b, 0):
                                scale = max(
                                    abs(np.mean(emission_a)),
                                    abs(np.mean(emission_b)),
                                    1e-12,
                                )

                                normalized_difference = (
                                    abs(np.mean(emission_a) - np.mean(emission_b)) / scale
                                )

                                emission_similarity = float(
                                    np.clip(
                                        1.0 - normalized_difference,
                                        0.0,
                                        1.0,
                                    )
                                )
                            else:
                                emission_similarity = 0.0

                            # Similarity in changes can identify miners that
                            # rise and fall at the same time.
                            if len(emission_a) >= 4:
                                emission_change_a = np.diff(emission_a)
                                emission_change_b = np.diff(emission_b)

                                if not np.isclose(
                                    np.std(emission_change_a),
                                    0,
                                ) and not np.isclose(
                                    np.std(emission_change_b),
                                    0,
                                ):
                                    change_correlation = float(
                                        np.corrcoef(
                                            emission_change_a,
                                            emission_change_b,
                                        )[0, 1]
                                    )

                                    emission_change_similarity = float(
                                        np.clip(
                                            (change_correlation + 1.0) / 2.0,
                                            0.0,
                                            1.0,
                                        )
                                    )
                                elif np.isclose(
                                    np.std(emission_change_a),
                                    0,
                                ) and np.isclose(
                                    np.std(emission_change_b),
                                    0,
                                ):
                                    emission_change_similarity = float(
                                        np.allclose(
                                            emission_change_a,
                                            emission_change_b,
                                            rtol=1e-5,
                                            atol=1e-8,
                                        )
                                    )
                                else:
                                    emission_change_similarity = 0.0

                    # ---------------------------------------------------
                    # 3. LastUpdate synchronization
                    # ---------------------------------------------------

                    update_synchronization = self._synchronized_update_score(
                        group_a,
                        group_b,
                    )

                    # ---------------------------------------------------
                    # 4. Validator-support overlap
                    # ---------------------------------------------------

                    validator_vector_a = validator_vectors.get(
                        miner_a,
                        np.array([]),
                    )

                    validator_vector_b = validator_vectors.get(
                        miner_b,
                        np.array([]),
                    )

                    validator_overlap = self._validator_overlap(
                        validator_vector_a,
                        validator_vector_b,
                    )

                    validator_weight_similarity = np.nan

                    validator_vector_size = min(
                        validator_vector_a.size,
                        validator_vector_b.size,
                    )

                    if validator_vector_size > 0:
                        truncated_a = validator_vector_a[:validator_vector_size].astype(
                            float
                        )

                        truncated_b = validator_vector_b[:validator_vector_size].astype(
                            float
                        )

                        finite_mask = np.isfinite(truncated_a) & np.isfinite(truncated_b)

                        truncated_a = truncated_a[finite_mask]
                        truncated_b = truncated_b[finite_mask]

                        if truncated_a.size > 0:
                            norm_a = np.linalg.norm(truncated_a)
                            norm_b = np.linalg.norm(truncated_b)

                            if norm_a > 0 and norm_b > 0:
                                cosine_value = float(
                                    np.dot(
                                        truncated_a,
                                        truncated_b,
                                    )
                                    / (norm_a * norm_b)
                                )

                                validator_weight_similarity = float(
                                    np.clip(
                                        cosine_value,
                                        0.0,
                                        1.0,
                                    )
                                )
                            elif norm_a == 0 and norm_b == 0:
                                validator_weight_similarity = 1.0
                            else:
                                validator_weight_similarity = 0.0

                    # ---------------------------------------------------
                    # 5. Active-state synchronization
                    # ---------------------------------------------------

                    active_synchronization = np.nan

                    if "active_a" in merged.columns and "active_b" in merged.columns:
                        active_frame = (
                            merged[["active_a", "active_b"]]
                            .apply(
                                pd.to_numeric,
                                errors="coerce",
                            )
                            .dropna()
                        )

                        if len(active_frame) >= 3:
                            active_synchronization = float(
                                np.mean(
                                    active_frame["active_a"].to_numpy()
                                    == active_frame["active_b"].to_numpy()
                                )
                            )

                    # ---------------------------------------------------
                    # 6. Composite pair risk
                    # ---------------------------------------------------

                    components = {
                        "trajectory_similarity": trajectory_similarity,
                        "emission_similarity": emission_similarity,
                        "emission_change_similarity": (emission_change_similarity),
                        "update_synchronization": (update_synchronization),
                        "validator_overlap": validator_overlap,
                        "validator_weight_similarity": (validator_weight_similarity),
                        "active_synchronization": (active_synchronization),
                    }

                    component_weights = {
                        "trajectory_similarity": 0.25,
                        "emission_similarity": 0.15,
                        "emission_change_similarity": 0.10,
                        "update_synchronization": 0.15,
                        "validator_overlap": 0.15,
                        "validator_weight_similarity": 0.15,
                        "active_synchronization": 0.05,
                    }

                    available_weight = sum(
                        component_weights[name]
                        for name, value in components.items()
                        if np.isfinite(value)
                    )

                    if available_weight > 0:
                        sybil_pair_score = float(
                            sum(
                                component_weights[name] * value
                                for name, value in components.items()
                                if np.isfinite(value)
                            )
                            / available_weight
                        )
                    else:
                        sybil_pair_score = np.nan

                    # Require enough independent evidence.
                    supporting_component_count = int(
                        sum(
                            np.isfinite(value) and value >= 0.80
                            for value in components.values()
                        )
                    )

                    is_sybil_candidate = bool(
                        np.isfinite(sybil_pair_score)
                        and sybil_pair_score >= self.sybil_pair_threshold
                        and supporting_component_count >= 2
                    )

                    pair_row = {
                        "netuid": int(netuid),
                        "miner_a": miner_a,
                        "miner_b": miner_b,
                        "overlapping_observations": int(len(merged)),
                        "trajectory_similarity": (trajectory_similarity),
                        "emission_similarity": emission_similarity,
                        "emission_correlation": emission_correlation,
                        "emission_change_similarity": (emission_change_similarity),
                        "update_synchronization": (update_synchronization),
                        "validator_overlap": validator_overlap,
                        "validator_weight_similarity": (validator_weight_similarity),
                        "active_synchronization": (active_synchronization),
                        "supporting_component_count": (supporting_component_count),
                        "sybil_pair_score": sybil_pair_score,
                        "is_sybil_candidate": int(is_sybil_candidate),
                    }

                    # Include individual trajectory similarities for inspection.
                    for feature, similarity in feature_similarity_values.items():
                        pair_row[f"{feature}_trajectory_similarity"] = similarity

                    pair_rows.append(pair_row)

                    if np.isfinite(sybil_pair_score):
                        miner_pair_scores[miner_a].append(sybil_pair_score)
                        miner_pair_scores[miner_b].append(sybil_pair_score)

                    if is_sybil_candidate:
                        candidate_neighbor_scores[miner_a].append(sybil_pair_score)
                        candidate_neighbor_scores[miner_b].append(sybil_pair_score)

                        candidate_graph.add_edge(
                            miner_a,
                            miner_b,
                            weight=sybil_pair_score,
                            trajectory_similarity=(trajectory_similarity),
                            validator_overlap=validator_overlap,
                            update_synchronization=(update_synchronization),
                        )

            # -----------------------------------------------------------
            # Construct candidate clusters from connected components
            # -----------------------------------------------------------

            connected_components = list(nx.connected_components(candidate_graph))

            valid_communities = [
                community
                for community in connected_components
                if len(community) >= self.minimum_sybil_cluster_size
                and candidate_graph.subgraph(community).number_of_edges() > 0
            ]

            # Sort clusters by size and then alphabetically for reproducibility.
            valid_communities = sorted(
                valid_communities,
                key=lambda community: (
                    -len(community),
                    sorted(community)[0],
                ),
            )

            miner_cluster_lookup: Dict[str, int] = {}

            for cluster_number, community in enumerate(
                valid_communities,
                start=1,
            ):
                cluster_graph = candidate_graph.subgraph(community).copy()

                edge_scores = [
                    data.get("weight", np.nan)
                    for _, _, data in cluster_graph.edges(data=True)
                ]

                edge_scores = [value for value in edge_scores if np.isfinite(value)]

                cluster_id = int(cluster_number)

                for miner_id in community:
                    miner_cluster_lookup[miner_id] = cluster_id

                cluster_emissions = (
                    subnet_group[subnet_group["miner_id"].isin(community)]
                    .groupby("block")["emission"]
                    .sum()
                )

                subnet_emissions = subnet_group.groupby("block")["emission"].sum()

                aligned_emissions = pd.concat(
                    [
                        cluster_emissions.rename("cluster_emission"),
                        subnet_emissions.rename("subnet_emission"),
                    ],
                    axis=1,
                ).fillna(0.0)

                cluster_share = aligned_emissions["cluster_emission"] / aligned_emissions[
                    "subnet_emission"
                ].replace(0, np.nan)

                cluster_rows.append(
                    {
                        "netuid": int(netuid),
                        "cluster_id": cluster_id,
                        "cluster_size": int(len(community)),
                        "miners": "|".join(sorted(community)),
                        "edge_count": int(cluster_graph.number_of_edges()),
                        "possible_edge_count": int(
                            len(community) * (len(community) - 1) / 2
                        ),
                        "cluster_density": float(nx.density(cluster_graph)),
                        "mean_pair_score": (
                            float(np.mean(edge_scores)) if edge_scores else np.nan
                        ),
                        "minimum_pair_score": (
                            float(np.min(edge_scores)) if edge_scores else np.nan
                        ),
                        "maximum_pair_score": (
                            float(np.max(edge_scores)) if edge_scores else np.nan
                        ),
                        "mean_cluster_emission_share": (
                            float(cluster_share.mean())
                            if cluster_share.notna().any()
                            else np.nan
                        ),
                        "maximum_cluster_emission_share": (
                            float(cluster_share.max())
                            if cluster_share.notna().any()
                            else np.nan
                        ),
                    }
                )

            # -----------------------------------------------------------
            # Produce miner-level risk records
            # -----------------------------------------------------------

            subnet_emission_total = subnet_group["emission"].sum()

            miner_emission_totals = subnet_group.groupby("miner_id")["emission"].sum()

            for miner_id in miner_ids:
                all_scores = miner_pair_scores.get(
                    miner_id,
                    [],
                )

                candidate_scores = candidate_neighbor_scores.get(
                    miner_id,
                    [],
                )

                miner_group = miner_groups[miner_id]

                miner_emission_total = float(miner_emission_totals.get(miner_id, 0.0))

                emission_share = (
                    miner_emission_total / subnet_emission_total
                    if subnet_emission_total > 0
                    else 0.0
                )

                maximum_pair_score = float(max(all_scores)) if all_scores else 0.0

                mean_pair_score = float(np.mean(all_scores)) if all_scores else 0.0

                maximum_candidate_score = (
                    float(max(candidate_scores)) if candidate_scores else 0.0
                )

                candidate_neighbor_count = int(candidate_graph.degree(miner_id))

                local_cluster_density = 0.0

                if miner_id in miner_cluster_lookup:
                    cluster_id = miner_cluster_lookup[miner_id]

                    community = valid_communities[cluster_id - 1]

                    local_cluster_density = float(
                        nx.density(candidate_graph.subgraph(community))
                    )

                miner_risk_rows.append(
                    {
                        "netuid": int(netuid),
                        "miner_id": miner_id,
                        "observation_count": int(len(miner_group)),
                        "miner_emission_total": (miner_emission_total),
                        "miner_emission_share": emission_share,
                        "maximum_pair_score": (maximum_pair_score),
                        "mean_pair_score": mean_pair_score,
                        "maximum_candidate_score": (maximum_candidate_score),
                        "high_similarity_neighbor_count": (candidate_neighbor_count),
                        "sybil_cluster_id": (
                            miner_cluster_lookup.get(
                                miner_id,
                                np.nan,
                            )
                        ),
                        "local_cluster_density": (local_cluster_density),
                        "belongs_to_sybil_cluster": int(miner_id in miner_cluster_lookup),
                    }
                )

        # ---------------------------------------------------------------
        # Convert records to DataFrames
        # ---------------------------------------------------------------

        pair_columns = [
            "netuid",
            "miner_a",
            "miner_b",
            "overlapping_observations",
            "trajectory_similarity",
            "emission_similarity",
            "emission_correlation",
            "emission_change_similarity",
            "update_synchronization",
            "validator_overlap",
            "validator_weight_similarity",
            "active_synchronization",
            "supporting_component_count",
            "sybil_pair_score",
            "is_sybil_candidate",
        ]

        cluster_columns = [
            "netuid",
            "cluster_id",
            "cluster_size",
            "miners",
            "edge_count",
            "possible_edge_count",
            "cluster_density",
            "mean_pair_score",
            "minimum_pair_score",
            "maximum_pair_score",
            "mean_cluster_emission_share",
            "maximum_cluster_emission_share",
        ]

        miner_risk_columns = [
            "netuid",
            "miner_id",
            "observation_count",
            "miner_emission_total",
            "miner_emission_share",
            "maximum_pair_score",
            "mean_pair_score",
            "maximum_candidate_score",
            "high_similarity_neighbor_count",
            "sybil_cluster_id",
            "local_cluster_density",
            "belongs_to_sybil_cluster",
        ]

        self.sybil_pairs_df = pd.DataFrame(pair_rows)

        if self.sybil_pairs_df.empty:
            self.sybil_pairs_df = pd.DataFrame(columns=pair_columns)
        else:
            self.sybil_pairs_df = self.sybil_pairs_df.sort_values(
                [
                    "is_sybil_candidate",
                    "sybil_pair_score",
                ],
                ascending=[False, False],
            ).reset_index(drop=True)

        self.sybil_clusters_df = pd.DataFrame(cluster_rows)

        if self.sybil_clusters_df.empty:
            self.sybil_clusters_df = pd.DataFrame(columns=cluster_columns)
        else:
            self.sybil_clusters_df = self.sybil_clusters_df.sort_values(
                [
                    "mean_pair_score",
                    "cluster_size",
                ],
                ascending=[False, False],
            ).reset_index(drop=True)

        self.miner_risk_df = pd.DataFrame(miner_risk_rows)

        if self.miner_risk_df.empty:
            self.miner_risk_df = pd.DataFrame(columns=miner_risk_columns)

            return (
                self.sybil_pairs_df,
                self.sybil_clusters_df,
            )

        # ---------------------------------------------------------------
        # Merge temporal miner-behavior metrics
        # ---------------------------------------------------------------

        behavior_df = self.build_miner_behavior_metrics()

        if not behavior_df.empty:
            self.miner_risk_df = self.miner_risk_df.merge(
                behavior_df,
                on=["netuid", "miner_id"],
                how="left",
            )

        # ---------------------------------------------------------------
        # Normalize miner-level risk components within each subnet
        # ---------------------------------------------------------------

        risk_component_columns = [
            "maximum_pair_score",
            "high_similarity_neighbor_count",
            "local_cluster_density",
            "miner_emission_share",
            "spike_rate",
            "emission_consensus_divergence",
        ]

        for column in risk_component_columns:
            if column not in self.miner_risk_df.columns:
                self.miner_risk_df[column] = np.nan

            self.miner_risk_df[f"scaled_{column}"] = self.miner_risk_df.groupby(
                "netuid",
                sort=False,
            )[column].transform(self._minmax_series)

        self.miner_risk_df["miner_sybil_risk_score"] = np.clip(
            0.40 * self.miner_risk_df["maximum_pair_score"].fillna(0)
            + 0.15 * self.miner_risk_df["scaled_high_similarity_neighbor_count"].fillna(0)
            + 0.15 * self.miner_risk_df["scaled_local_cluster_density"].fillna(0)
            + 0.10 * self.miner_risk_df["scaled_miner_emission_share"].fillna(0)
            + 0.10 * self.miner_risk_df["scaled_spike_rate"].fillna(0)
            + 0.10 * self.miner_risk_df["scaled_emission_consensus_divergence"].fillna(0),
            0.0,
            1.0,
        )

        # Risk labels are descriptive screening categories.
        self.miner_risk_df["sybil_risk_category"] = pd.cut(
            self.miner_risk_df["miner_sybil_risk_score"],
            bins=[
                -np.inf,
                0.25,
                0.50,
                0.75,
                np.inf,
            ],
            labels=[
                "low",
                "moderate",
                "high",
                "critical",
            ],
        )

        self.miner_risk_df = self.miner_risk_df.sort_values(
            [
                "miner_sybil_risk_score",
                "maximum_pair_score",
            ],
            ascending=[False, False],
        ).reset_index(drop=True)

        return (
            self.sybil_pairs_df,
            self.sybil_clusters_df,
        )
