"""All analyses, in the order they run. Each one's questions and tables are listed on the class."""
from .base import Analysis
from .checks import DataChecks
from .exploits import ExploitAnalysis
from .insights import InsightAnalysis
from .rewards import RewardAnomalyAnalysis
from .scoring import ScoringAnalysis
from .sufficiency import SufficiencyAnalysis
from .sybil import SybilAnalysis
from .telemetry import TelemetryAnalysis

# order matters: later analyses read earlier results
ALL_ANALYSES = [DataChecks, ExploitAnalysis, ScoringAnalysis, SybilAnalysis, RewardAnomalyAnalysis,
                TelemetryAnalysis, SufficiencyAnalysis, InsightAnalysis]
BY_KEY = {a.key: a for a in ALL_ANALYSES}
# the five focus-area problems that get ranked
PROBLEM_ANALYSES = [ExploitAnalysis, TelemetryAnalysis, ScoringAnalysis, InsightAnalysis, SybilAnalysis]

__all__ = ["Analysis", "ALL_ANALYSES", "BY_KEY", "PROBLEM_ANALYSES"] + [a.__name__ for a in ALL_ANALYSES]
