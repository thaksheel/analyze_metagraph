import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from modules.analyzer import Analyzer
from modules.analyzer.config import *

# ---------------------------------------------------------------- settings
BLOCKS = "./exports/blocks/"  # collector output (folder of block_*.json.gz, or one merged .json)
OUT_DIR = "./exports"  # cache + results
ONLY = [
    "exploits",
    "scoring",
    "sybil",
    "rewards",
    "telemetry",
    "sufficiency",
    "insights",
]
# ---------------------------------------------------------------------------
all_config = Config()
exploit_config = ExploitConfig(
    cp_min_segment_days=2,
    cp_penalty_scale=3,
    cp_min_effect=1,
    incident_window_days=3,
    capture_min_share=0.15,
    cohort_capture_fraction=0.5,
    cohort_max_fraction=0.2,
    recovery_unwind=0.75,
    recovery_consecutive=3,
    burn_rise_threshold=0.05,
)

az = Analyzer(blocks=BLOCKS, out_dir=OUT_DIR, config=all_config)
az.config.paths.results_dir = "./exports/w44"

all_results = az.run_all(ONLY, save=False)
exploits_results = az.run_exploits(exploit_config=exploit_config, save=True)


if "ranking" in all_results.analyses():
    print("\nRanking")
    print(
        all_results.ranking.problems[
            ["rank", "problem", "priority", "size_value", "impact_pct"]
        ].to_string(index=False)
    )
print(
    f"\nTables saved to {az.config.paths.results_dir} (see _catalog.csv and summary.md)"
)
print("END")
