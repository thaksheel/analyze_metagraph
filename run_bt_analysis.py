import os
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 

from modules.analyzer.config import Config
from modules.analyzer.pipeline import run


cfg = Config(
    raw_snapshots="./exports/bss500.json",
    identity_file="",
    processed_dir=os.path.join("./exports", "processed"),
    results_dir=os.path.join("./exports", "results"),
)
out = run(cfg, skip_prepare=False)


print("END")
