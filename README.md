# Bittensor Metagraph Information Collector

A tool for collecting and analyzing metagraph information from the Bittensor network across different subnets and block heights.

## Overview

This project collects specific datapoints from Bittensor subnets at different block ages to enable analysis of subnet behaviors and network dynamics over time. The goal is to extract metagraph data in a structured format suitable for data analysis and behavioral insights.

## Current Status

⚠️ **Work in Progress** — `run.py` is currently in active development and testing phase. The core functionality works, but the code is not yet packaged into a reusable class for broader use.

### Dataset Fields

A dataset class has been outlined with specific fields to capture from the Bittensor metagraph. These fields are currently being tested against the `query_map` function to ensure reliable data collection.

## Active Issues Being Resolved 

1. **Archive Mode for Older Blocks**
   - Need to implement archive mode queries to fetch historical blockchain data
   - Currently exploring Bittensor's archive capabilities

2. **Batch Processing for Large-Scale Data Collection**
   - Fetching large numbers of blocks currently requires 100+ hours of sequential queries
   - Working on implementing efficient batch processing to significantly reduce runtime
   - Investigating async patterns and query optimization

3. **Data Packaging and Array Organization**
   - Developing a consolidated nd-array structure for efficient data storage and analysis
   - Ensuring data is organized for easy consumption in data analysis workflows

## Project Structure

```
├── run.py              # Main data collection script (WIP)
├── src/
│   ├── metagraph_fetch.py
│   └── utils.py
├── exports/            # Output data directory
└── temp/               # Temporary files
```

## Goals

- Collect granular subnet-specific datapoints across different time periods
- Enable behavioral analysis of subnets through metagraph metrics
- Create efficient pipelines for large-scale blockchain data collection
- Package results in formats suitable for statistical and machine learning analysis

## Next Steps

1. Finalize and validate all dataset fields via `query_map`
2. Implement archive mode support for historical data
3. Develop batch processing pipeline with async optimization
4. Refactor into a production-ready class structure
5. Create comprehensive data export formats

## Notes

Once the code stabilizes and the core issues are resolved, this will be refactored into a proper, reusable class that can be easily integrated into other projects or used directly for data collection workflows.
