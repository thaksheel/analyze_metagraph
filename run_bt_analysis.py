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
    cp_min_segment_days=0.5,  # bursts can last hours, so allow change points closer together
    cp_penalty_scale=1.5,  # lower penalty makes short shifts detectable
    cp_min_effect=0.75,  # slightly more sensitive to smaller jumps
    incident_window_days=1.0,  # a 3-day window dilutes a burn that lasts a few hours
    capture_min_share=0.50,  # --< catch cohorts taking a smaller share of emission per event
    cohort_capture_fraction=0.6,  # cohort must cover more of the gains, so it reflects the core group
    cohort_max_fraction=0.10,  # --< "small group" means at most ~10% of miners
    recovery_unwind=0.5,  # colluders often unwind only part of the gain
    recovery_consecutive=3,  # keep as is, but raise to 5 if your snapshots are very frequent
    burn_rise_threshold=0.03,  # more sensitive to burn-side redirection
)

az = Analyzer(blocks=BLOCKS, out_dir=OUT_DIR, config=all_config)
az.config.paths.results_dir = "./exports/w44"

cohort_max_fractions = [0.01, 0.02, 0.04, 0.08, 0.2]
capture_min_share = [0.4, 0.5, 0.6, 0.7, 0.8]
for f in cohort_max_fractions: 
    exploit_config.cohort_max_fraction = f 
    exploits_results = az.run_exploits(exploit_config=exploit_config, save=False)
    cols, headline = az.exploit_aggregated_report(exploits_results)
    with pd.ExcelWriter(f"./exports/w44/exploits_agg_cmf_{f}.xlsx", engine="openpyxl") as writer:
        cols.to_excel(writer, sheet_name="features", index=False)
        headline.to_excel(writer, sheet_name="summ", index=False)

for f in capture_min_share: 
    exploit_config.capture_min_share = f 
    exploits_results = az.run_exploits(exploit_config=exploit_config, save=False)
    cols, headline = az.exploit_aggregated_report(exploits_results)
    with pd.ExcelWriter(f"./exports/w44/exploits_agg_cms_{f}.xlsx", engine="openpyxl") as writer:
        cols.to_excel(writer, sheet_name="features", index=False)
        headline.to_excel(writer, sheet_name="summ", index=False)

# all_results = az.run_all(ONLY, save=False)
# if "ranking" in all_results.analyses():
#     print("\nRanking")
#     print(
#         all_results.ranking.problems[
#             ["rank", "problem", "priority", "size_value", "impact_pct"]
#         ].to_string(index=False)
#     )
# print(
#     f"\nTables saved to {az.config.paths.results_dir} (see _catalog.csv and summary.md)"
# )

print("END")
