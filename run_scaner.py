import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 

from src import MetagraphScanner, BittensorTemporalAnalyzer
from src.metagraph_fetch import MetagraphManager

mm = MetagraphManager()
ms = MetagraphScanner()

bss = mm.load_blocksnapshots(inpath="./exports/bss1.json")
bis = mm.load_cache_block_info(bi_path="./exports/block_infos.json")

bta = BittensorTemporalAnalyzer(
    snapshots=bss, 
    output_directory="./exports/", 
)
miner_behave_out = bta.build_miner_behavior_metrics()
telemetry_out = bta.evaluate_telemetry_sufficiency()
sybil_out = bta.detect_sybil_candidates()


print("END")
