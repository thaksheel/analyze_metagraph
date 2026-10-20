"""Dataset: all subnets of one collection.

    ds = Dataset(cfg)            # builds the cache on first use
    ds.netuids                   # subnets in the data
    sn = ds[20]                  # Subnet for netuid 20
    for sn in ds: ...            # iterate (one subnet in memory at a time)
"""
import json
import os
from typing import Iterator, List

from ..config import Config
from .loader import BlockLoader
from .subnet import Subnet


class Dataset:
    def __init__(self, cfg: Config, rebuild: bool = False, verbose: bool = True):
        self.cfg = cfg
        self.loader = BlockLoader(cfg.paths.blocks, cfg.paths.cache_dir)
        self.meta = self.loader.build(force=rebuild, verbose=verbose)
        self.netuids: List[int] = self.meta["netuids"]

    def __len__(self) -> int:
        return len(self.netuids)

    def __getitem__(self, netuid: int) -> Subnet:
        if netuid not in self.netuids:
            raise KeyError(f"netuid {netuid} not in data; available: {self.netuids}")
        return Subnet(netuid, os.path.join(self.cfg.paths.cache_dir, f"subnet_{netuid}.npz"))

    def __iter__(self) -> Iterator[Subnet]:
        for n in self.netuids:
            yield self[n]

    def __repr__(self):
        return f"Dataset({len(self)} subnets, {len(self.meta['blocks'])} blocks, source={self.cfg.paths.blocks})"
