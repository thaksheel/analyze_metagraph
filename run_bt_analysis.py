import os
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 

from modules.analyzer import Analyzer

# ---------------------------------------------------------------- settings
BLOCKS = "./exports/blocks/"  # collector output (folder of block_*.json.gz, or one merged .json)
OUT_DIR = "./exports"  # cache + results
ONLY = None  # None = everything, or e.g. ["sybil", "rewards"] (dependencies added)
OVERRIDES = {  # change any setting: "section.name": value
    # "sybil.edge_min": 0.7,
    # "exploits.capture_min_share": 0.2,
}
FIND = ""  # set to a word (e.g. "churn") to search instead of running
# ---------------------------------------------------------------------------

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 90)

az = Analyzer(BLOCKS, out_dir=OUT_DIR)
for path, value in OVERRIDES.items():
    section, name = path.split(".")
    setattr(getattr(az.cfg, section), name, value)

if FIND:
    print(az.find(FIND).to_string(index=False))
else:
    res = az.run(ONLY)
    if "ranking" in res.analyses():
        print("\nRanking")
        print(
            res.ranking.problems[
                ["rank", "problem", "priority", "size_value", "impact_pct"]
            ].to_string(index=False)
        )
    print(
        f"\nTables saved to {az.cfg.paths.results_dir} (see _catalog.csv and summary.md)"
    )

print("END")