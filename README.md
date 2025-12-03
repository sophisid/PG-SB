# PG-Schema-Bench: A Benchmark for Schema Discovery in Property Graphs
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17801336.svg)](https://doi.org/10.5281/zenodo.17801336)

A configurable benchmarking system for evaluating schema discovery systems on Property Graphs.


## **1. Repository Structure**

```
PG_bench/
│
├── benchmark.py               # main benchmark runner
├── benchmark_config.json      # configuration for paths, Neo4j versions, noise levels
├── benchmark_commands.json    # algorithms or scripts to run at each noise level
│
├── datasets/                  # dataset folders (CSV + metadata – NO DUMPS)
│   ├── fib25/
│   ├── mb6/
│   ├── hetio/
│   ├── icij/
│   ├── cord19/
│   └── ...
│
└── output/                    # results generated automatically
```

---

## **2. Downloading the Neo4j Dumps**

Neo4j dumps **are not stored in this repository** because of GitHub size limits.

Instead:

### **Download all dataset dumps from Zenodo:**

**Zenodo link:**
[https://zenodo.org/records/17801336](https://zenodo.org/records/17801336)

After downloading, place each dataset dump inside its folder:

```
./datasets/<dataset_name>/
```

For example:

```
datasets/fib25/fib25-neo4j-4.4.0.dump
datasets/pole/pole-neo4j-4.4.0.dump
datasets/iyp/iyp-neo4j-5.25.1.dump
```

---

## **3. Downloading Neo4j Community Editions**

Different datasets use different Neo4j versions (e.g., `4.4.0`, `5.1.0`, `5.25.1`).
These cannot be stored in GitHub either.

Download Neo4j Community Edition from the official website:

**Neo4j Download Center:**
[https://neo4j.com/download-center/](https://neo4j.com/download-center/)

After downloading, extract them somewhere on your machine, e.g.:

```
./neo4j-community-4.4.0/
./neo4j-community-5.1.0/
./neo4j-community-5.25.1/
```

Then reference these paths in the **benchmark_config.json**.

---

## **4. Configuration (benchmark_config.json)**

This file defines *all* paths, Neo4j versions, and benchmark settings.

Example:

```json
{
  "datasets_dir": "./datasets",
  "output_dir": "./output",
  "commands_file": "./benchmark_commands.json",
  "neo4j_password": "password",
  "neo4j_port": 7687,

  "noise_levels": [0, 10, 20, 30, 40],
  "label_percents": [0.0, 0.5, 1.0],

  "neo4j_dirs": {
    "4.4.0": "./neo4j-community-4.4.0",
    "5.1.0": "./neo4j-community-5.1.0",
    "5.25.1": "./neo4j-community-5.25.1"
  }
}
```

## **5. Defining Algorithms (benchmark_commands.json)**

Users define *their own* algorithms/scripts to run during the benchmark.

Example:

```json
[
  {
    "name": "LSH_MINHASH",
    "cwd": "./schemadiscovery",
    "cmd": "sbt \"run LSH MINHASH\""
  },
  {
    "name": "ELSH",
    "cwd": "./schemadiscovery",
    "cmd": "sbt \"run LSH\""
  },
  {
    "name": "KMEANS",
    "cwd": "./schemadiscovery",
    "cmd": "sbt \"run KMEANS\""
  },
  {
    "name": "PG",
    "cwd": "./pg-schemainference",
    "cmd": "python3 cluster_script.py"
  }
]
```

* Add/remove entries freely
* You can run Python, Scala, Java, shell commands — anything.
* Each run automatically produces a log file in `output/<dataset>/`.

---

## **6. Running the Benchmark**

```bash
python3 benchmark.py --config ./config.json
```

The script performs:

1. Detect dataset dump
2. Match proper Neo4j version
3. Stop Neo4j
4. Load clean dump
5. Apply noise
6. Remove labels
7. Run all user-defined commands
8. Save logs
9. Repeat for next noise level & dataset

---

## **7. Dataset CSV Metadata**

Inside each dataset folder has:

```
node_properties.csv
edge_properties.csv
edge_labels.csv
node_labels.csv
```

These define which properties/labels/noise to apply.

Each CSV contains *one value per line* (any single column is accepted).

---

## **8. Artifacts**

The datasets used and their sources are listed as:

* **POLE**
  Source: Neo4j Graph Examples
  GitHub: [https://github.com/neo4j-graph-examples/pole](https://github.com/neo4j-graph-examples/pole)

* **MB6**
  Source: Connectome (Takemura et al.)
  CSV Dataset: [https://github.com/sophisid/PG-HIVE/tree/master/datasets/MB6](https://github.com/sophisid/PG-HIVE/tree/master/datasets/MB6)

* **HET.IO**
  Source: Himmelstein et al.
  GitHub: [https://github.com/hetio/hetionet](https://github.com/hetio/hetionet)

* **FIB25**
  Source: Takemura et al.
  CSV Dataset: [https://github.com/sophisid/PG-HIVE/tree/master/datasets/FIB25](https://github.com/sophisid/PG-HIVE/tree/master/datasets/FIB25)

* **ICIJ (Offshore Leaks)**
  GitHub: [https://github.com/ICIJ/offshoreleaks-data-packages](https://github.com/ICIJ/offshoreleaks-data-packages)

* **LDBC**
  Source: LDBC Benchmark
  CSV Dataset: [https://github.com/sophisid/PG-HIVE/tree/master/datasets/LDBC](https://github.com/sophisid/PG-HIVE/tree/master/datasets/LDBC)

* **CORD-19**
  Source: AllenAI
  GitHub: [https://github.com/allenai/cord19](https://github.com/allenai/cord19)

* **IYP – Internet Yellow Pages**
  Source: Fontugne et al.
  GitHub: [https://github.com/InternetHealthReport/internet-yellow-pages](https://github.com/InternetHealthReport/internet-yellow-pages)

---

## **9. Contributions**

Contributions are very welcome.

You can contribute by:

* Adding new datasets
* Extending/Improving the benchmark pipeline
* Reporting issues or submitting pull requests
