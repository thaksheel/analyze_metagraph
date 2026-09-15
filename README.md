# Bittensor Metagraph Information Collector

This project collects metagraph information from the Bittensor network for subnet-level analysis across different block ages and network conditions. The goal is to gather structured data points from multiple subnets so that subnet behavior can be studied at both the individual and aggregate level.

## Overview

The repository is designed to collect metagraph and subnet state information from Bittensor and store it in a format that is easy to analyze with Python-based scientific and statistical workflows. The main workflow now relies on the `MetagraphManager` class in `src/metagraph_fetch.py`, which provides the core functionality for retrieving block information, selecting subnets by metric, and collecting metagraph snapshots at specific blocks.

## Current Status

Most of the main implementation issues have been resolved, and the project is now using the `MetagraphManager` workflow to collect the information needed for analysis. The code is no longer limited to a single exploratory script; it is structured around a reusable class for data collection and caching.

The remaining work is focused on scaling the collection process and stabilizing edge cases around network requests and large data retrieval.

## Recommended Workflow

For someone replicating this repo, the intended process is:

1. Run `run_cache.py` first to collect the block information needed for the analysis.
2. Run `run.py` to collect metagraph data for a selected set of netuids.
3. Use the exported cache and snapshot files in `exports/` for downstream analysis.

This sequence ensures that block metadata is available before metagraph snapshots are gathered for a given set of network conditions.

## Data Collection Targets

The long-term objective is to collect a large set of subnet observations over time, including:

- multiple subnets
- different block ages
- repeated snapshots across time windows
- aggregated metrics across many subnet states

The immediate analytic goal is to collect approximately 500 blocks of data for a specific subset of netuids, with at least 30 subnets included in the collection set. This is intended to support modeling subnet behavior at an aggregate level and analyzing how different subnet conditions evolve over time.

## Dataset and Field Scope

A dataset structure has been defined to capture the needed metagraph fields, and the project is using those fields in the collection flow. Some of the field mappings are still being validated against the underlying `query_map` calls, but the general framework is in place and the collection pipeline is working.

The broader objective is to organize the final output into a concise array-based dataset or structured tabular representation for efficient analysis and modeling.


## Project Structure

```
├── run.py              # Main metagraph collection workflow
├── run_cache.py        # Block metadata collection workflow
├── src/
│   ├── metagraph_fetch.py   # MetagraphManager implementation
│   └── utils.py            # Utility helpers
├── exports/            # Cached block and snapshot data
├── temp/               # Temporary working files
├── README.md           # Project overview and workflow notes
└── ...
```

## Goals

- Collect subnet-specific datapoints across multiple block windows
- Study aggregate behavior across many subnets over time
- Build a scalable dataset for behavior analysis and modeling
- Package the resulting information into a concise, analysis-ready data structure

## Next Steps

1. Continue validating the metagraph field set used in the collection pipeline
2. Expand the batch collection workflow for larger runs
3. Collect 500 blocks of data for a focused subnet set, aiming for at least 30 subnets
4. Improve handling of RPC request limits and failure recovery
5. Package the final dataset into a stable structure for downstream modeling and analysis

## Notes

This project is still evolving, but the core collection workflow is now in place. The current emphasis is on scaling the data collection process, improving historical block retrieval, and producing a consistent dataset suitable for subnet behavior analysis.
