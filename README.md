# PG-Schema-Bench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17801335.svg)](https://doi.org/10.5281/zenodo.17801335)

Benchmarking utilities for schema discovery on property graphs, plus scripts for exporting graph statistics.

## Overview

This repository does three main things:

- loads Neo4j dump files for multiple datasets
- applies benchmark perturbations such as property noise and label removal
- runs one or more schema discovery approaches on every benchmark case

Main scripts:

- `run.sh`: installs Python requirements and starts the benchmark
- `benchmark.py`: main benchmark runner
- `evaluation.py`: computes evaluation metrics from exported CSVs
- `stats.py`: exports graph statistics from a running Neo4j database
- `cp_metrics.py`: computes CP metrics

Main folders:

- `datasets/`: dataset folders with metadata CSVs and Neo4j dump files
- `output/`: benchmark logs and method outputs
- `stats_*`: precomputed statistics for several datasets

## Download the dumps first

Before running the benchmark, download the dataset dump files from Zenodo:

- https://zenodo.org/records/17801336

After downloading them, place each dump inside the matching dataset folder under `datasets/`.

Examples:

```text
datasets/star-wars/star-wars-neo4j-4.4.0.dump
datasets/fib25/fib25-neo4j-4.4.0.dump
datasets/icij/icij-neo4j-4.4.0.dump
datasets/iyp/iyp-neo4j-5.25.1.dump
```

The benchmark expects the dump to already be in the correct dataset directory before it starts.

## Dataset layout

Each dataset folder under `datasets/` is expected to contain:

- a dump named like `*neo4j-X.Y.Z.dump`
- `node_properties.csv`
- `edge_properties.csv`
- `node_labels.csv`
- optionally `edge_labels.csv`


The benchmark detects the Neo4j version directly from the dump filename.

Example:

```text
datasets/fib25/fib25-neo4j-4.4.0.dump
datasets/iyp/iyp-neo4j-5.25.1.dump
```

## Neo4j modes

The benchmark supports two Neo4j runtime modes.

### Community mode

Use `neo4j_mode: "community"` if you want this repo to manage Neo4j installs directly.

Behavior:

- uses a configured Community install if it already exists
- downloads Neo4j Community automatically if it is missing and `neo4j_auto_download` is `true`
- updates `conf/neo4j.conf` with the configured memory settings
- loads the dump with `neo4j-admin`

### Desktop mode

Use `neo4j_mode: "desktop"` if you want to reuse a DBMS managed by Neo4j Desktop.

Important:

- point `neo4j_desktop_dirs` to the actual DBMS home directory, not the Desktop app
- the benchmark then uses that DBMS's `bin/neo4j`, `bin/neo4j-admin`, and `bin/cypher-shell`
- dump loading is still automated; you do not need to import the dump manually through the Desktop UI each time

## Configuration

The benchmark reads [config.json](/Users/sophia-euthymiasideris/Desktop/PG_bench/config.json). A template is available in [config_template.json](/Users/sophia-euthymiasideris/Desktop/PG_bench/config_template.json).

Current example:

```json
{
  "datasets_dir": "./datasets",
  "output_dir": "./output",
  "commands_file": "./benchmark_commands.json",
  "neo4j_password": "password",
  "neo4j_port": 7687,
  "noise_levels": [0, 10, 20, 30, 40],
  "label_percents": [0.0, 0.5, 1.0],
  "neo4j_mode": "community",
  "neo4j_dirs": {
    "4.4.0": "./neo4j-community-4.4.0",
    "5.1.0": "./neo4j-community-5.1.0",
    "5.25.1": "./neo4j-community-5.25.1"
  },
  "neo4j_desktop_dirs": {},
  "neo4j_auto_download": true,
  "neo4j_download_dir": "./neo4j_runtimes",
  "neo4j_memory": {
    "heap_initial": "2G",
    "heap_max": "4G",
    "pagecache": "2G"
  },
  "dataset_order": [
    "starwars",
    "pole",
    "mb6",
    "het",
    "fib",
    "icij",
    "cord",
    "twitch",
    "ldbc",
    "iyp"
  ],
  "run_external_commands": true,
  "query_batch_size": 10000
}
```

Main keys:

- `datasets_dir`: root directory for dataset folders
- `output_dir`: benchmark outputs and logs
- `commands_file`: methods to run
- `neo4j_password`: password used by `cypher-shell`
- `neo4j_port`: port used by Neo4j
- `noise_levels`: property-removal percentages
- `label_percents`: fraction of nodes that lose labels
- `neo4j_mode`: `community` or `desktop`
- `neo4j_dirs`: version-to-install-path mapping for Community mode
- `neo4j_desktop_dirs`: version-to-DBMS-home mapping for Desktop mode
- `neo4j_auto_download`: auto-download missing Community installs
- `neo4j_download_dir`: fallback directory for downloaded Community installs
- `neo4j_memory`: memory settings written into `neo4j.conf`
- `dataset_order`: benchmark execution order; aliases such as `starwars`, `het`, `fib`, and `cord` are normalized
- `run_external_commands`: whether schema discovery methods should actually run
- `query_batch_size`: batch size for the heavy Cypher write operations

## Benchmark commands

Schema discovery methods are defined in [benchmark_commands.json](/Users/sophia-euthymiasideris/Desktop/PG_bench/benchmark_commands.json).

Each command can contain:

- `name`: method label
- `cmd`: shell command to run
- `cwd`: working directory
- `repo`: optional GitHub repo to clone before running
- `clone_dir`: local checkout path
- `branch`: branch to clone or pull
- `update_existing`: whether to pull an existing checkout
- `setup_cmd`: optional dependency-install or build command, run once per checkout before dataset processing starts

Current methods:

- `PG_HIVE_LSH`
- `PG_HIVE_MINHASH`

## How the benchmark runs

Recommended entrypoint:

```bash
./run.sh
```

With evaluation:

```bash
./run.sh --eval
```

You can also run the Python script directly:

```bash
python3 benchmark.py --config ./config.json
```

Execution flow:

1. loads the config
2. reads `benchmark_commands.json`
3. clones or updates all external method repositories
4. runs each method's `setup_cmd` once
5. iterates through the datasets in `dataset_order`
6. detects the dump and required Neo4j version
7. resolves the Neo4j runtime
8. writes Neo4j memory settings into `neo4j.conf`
9. stops Neo4j if needed
10. loads the dump into the `neo4j` database
11. starts Neo4j and waits until `cypher-shell "RETURN 1"` succeeds
12. saves original labels
13. removes properties and labels according to the configured benchmark case
14. runs all configured methods
15. optionally runs evaluation if the expected CSVs exist and `--eval` was passed

## Important runtime behavior

- each noise level starts from a fresh dump load
- heavy write queries are batched to reduce memory usage
- external repos are prepared once at the beginning, not inside every dataset loop
- if `run_external_commands` is `false`, the benchmark still loads dumps and applies benchmark transformations, but skips the schema discovery methods

## Outputs

Outputs are written under `output/<dataset_name>/`.

Common files:

- `log_noise{noise}_*.txt`: benchmark-side logs for the Cypher mutation steps
- `output_{DATASET}_noise{noise}_labels{percent}_{METHOD}.txt`: stdout/stderr from each external method
- evaluation CSVs and metric outputs when evaluation is enabled

## Evaluation inputs

Evaluation runs only when:

- you pass `--eval`
- all four expected CSV files exist for a dataset / noise / label-percent / method combination

Expected files:

- `original_nodes_...csv`
- `predicted_nodes_...csv`
- `original_edges_...csv`
- `predicted_edges_...csv`

Expected columns:

`original_nodes.csv`

- `_nodeId`
- `original_label`

`predicted_nodes.csv`

- `merged_cluster_id`
- `sortedLabels`
- `nodeIdsInCluster`

`original_edges.csv`

- `srcId`
- `dstId`
- `relationshipType`
- `srcType`
- `dstType`

`predicted_edges.csv`

- `merged_cluster_id`
- `relationshipTypes`
- `srcLabels`
- `dstLabels`
- `edgeIdsInCluster`

## Graph statistics

`stats.py` exports a structural profile from a running Neo4j database.

Example:

```bash
python3 stats.py \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --password password \
  --db neo4j \
  --out stats_fib
```

Important outputs include:

- `summary_counts.csv`
- `summary_counts.json`
- `cp_operationalization.csv`
- `cp_operationalization.json`
- `node_label_counts.csv`
- `relationship_type_counts.csv`
- `node_type_counts.csv`
- `node_patterns.csv`
- `edge_type_counts.csv`
- `edge_patterns.csv`
- `profile.json`

## Troubleshooting

If Neo4j does not become ready:

- check `neo4j-community-<version>/logs/neo4j.log`
- make sure you are using Java 11 for Neo4j 4.4.x
- keep `neo4j_memory` realistic for your machine; too-large values cause startup failure

If a method repo fails during setup:

- check whether the method needs extra system tools such as `sbt`
- check the method log under `output/<dataset>/`

If large datasets stall:

- lower `query_batch_size`
- adjust `neo4j_memory`

# Note
For `ldbc-sbn`, you can the benchmark here: https://github.com/ldbc/ldbc_snb_datagen_hadoop