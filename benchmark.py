import os
import re
import glob
import time
import tempfile
import subprocess
import csv
import json
import argparse

DATASETS_BASE_DIR = None
OUTPUT_BASE_DIR = None
COMMANDS_FILE = None
NEO4J_PASSWORD = None
NEO4J_PORT = None
NOISE_LEVELS = None
LABEL_PERCENTS = None
NEO4J_DIRS = None


def run(cmd, cwd=None, log_file=None, check=True):
    print(f"[RUN] {cmd} (cwd={cwd})")
    if log_file:
        with open(log_file, "a") as f:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT
            )
    else:
        proc = subprocess.run(cmd, cwd=cwd, shell=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return proc.returncode


def detect_dump_and_version(dataset_dir: str):
    dumps = glob.glob(os.path.join(dataset_dir, "*neo4j*.dump"))
    if not dumps:
        raise FileNotFoundError(f"No dump file found in {dataset_dir}")

    dump_file = sorted(dumps)[0]
    base = os.path.basename(dump_file)

    # Robust regex for version X.Y.Z
    m = re.search(r"neo4j-(\d+\.\d+\.\d+)", base)
    if not m:
        raise ValueError(
            f"Could not detect Neo4j version from dump name: {dump_file}. "
            f"Expected something like '*neo4j-X.Y.Z.dump'."
        )

    version = m.group(1).strip()

    return dump_file, version

def neo4j_dir_for_version(version: str) -> str:
    global NEO4J_DIRS
    if version in NEO4J_DIRS:
        neo4j_dir = NEO4J_DIRS[version]
        if not os.path.isdir(neo4j_dir):
            raise FileNotFoundError(f"Configured Neo4j dir for version {version} does not exist: {neo4j_dir}")
        return neo4j_dir

    raise FileNotFoundError(
        f"No Neo4j directory configured for version {version}. "
        f"Add it under 'neo4j_dirs' in your benchmark_config.json."
    )


def neo4j_store_lock_path(neo4j_dir: str) -> str:
    candidates = [
        os.path.join(neo4j_dir, "data", "databases", "store_lock"),
        os.path.join(neo4j_dir, "data", "databases", "neo4j", "store_lock"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def stop_neo4j(neo4j_dir: str):
    global NEO4J_PORT
    print(f"Stopping Neo4j at {neo4j_dir}...")
    run(f"{neo4j_dir}/bin/neo4j stop", check=False)
    time.sleep(10)
    status_code = subprocess.run(
        f"{neo4j_dir}/bin/neo4j status",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    ).returncode
    if status_code == 0:
        raise RuntimeError("Neo4j failed to stop")

    # Kill any leftover process on Bolt port
    print("Killing any process on port", NEO4J_PORT)
    subprocess.run(f"lsof -t -i :{NEO4J_PORT} | xargs -r kill -9", shell=True)

    # Remove store_lock
    store_lock = neo4j_store_lock_path(neo4j_dir)
    if os.path.exists(store_lock):
        print(f"Removing store_lock {store_lock}")
        os.remove(store_lock)


def load_dump(neo4j_dir: str, dump_file: str, version: str):
    print(f"Loading dump {dump_file} into Neo4j {version}")
    if version.startswith("4."):
        cmd = f'{neo4j_dir}/bin/neo4j-admin load --from="{dump_file}" --database=neo4j --force'
    else:
        cmd = f'{neo4j_dir}/bin/neo4j-admin database load neo4j --from="{dump_file}" --overwrite-destination=true'
    run(cmd)


def start_neo4j_and_wait(neo4j_dir: str, timeout_s: int = 300):
    global NEO4J_PASSWORD
    print("Starting Neo4j...")
    run(f"{neo4j_dir}/bin/neo4j start", check=True)
    print("Waiting for Neo4j to be ready...")
    start = time.time()
    while True:
        rc = subprocess.run(
            f'{neo4j_dir}/bin/cypher-shell -u neo4j -p "{NEO4J_PASSWORD}" "RETURN 1"',
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ).returncode
        if rc == 0:
            print("Neo4j is up.")
            break
        if time.time() - start > timeout_s:
            raise TimeoutError("Neo4j failed to start within timeout")
        time.sleep(5)


def cypher(neo4j_dir: str, query: str, log_path: str = None):
    global NEO4J_PASSWORD
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cypher", delete=False) as tmp:
        tmp.write(query)
        tmp_path = tmp.name
    try:
        cmd = f'{neo4j_dir}/bin/cypher-shell -u neo4j -p "{NEO4J_PASSWORD}" -f "{tmp_path}" --debug'
        run(cmd, log_file=log_path)
    finally:
        os.remove(tmp_path)


def clean_spark_tmp():
    subprocess.run(
        'rm -rf /tmp/blockmgr-* /tmp/spark-* /tmp/**/temp_shuffle_* 2>/dev/null',
        shell=True
    )


def read_csv_list(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}")
        return []

    values = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            first = None
            for col in row:
                col = col.strip()
                if col:
                    first = col
                    break
            if not first:
                continue
            low = first.lower()
            if low in {"property", "properties", "label", "labels", "name", "relationship", "type"}:
                continue
            values.append(first)
    print(f"[INFO] Loaded {len(values)} items from {csv_path}")
    return values


def load_commands(path: str):
    """
    Διαβάζει ένα JSON με λίστα από:
    { "name": "...", "cwd": "...", "cmd": "..." }
    και το επιστρέφει ως λίστα dicts.
    Αν το αρχείο δεν υπάρχει, επιστρέφει [].
    """
    if not os.path.exists(path):
        print(f"[INFO] Commands file not found ({path}), no external commands will be run.")
        return []

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Commands file {path} must contain a JSON list")

    for idx, cmd in enumerate(data):
        if not isinstance(cmd, dict):
            raise ValueError(f"Command #{idx} is not an object")
        if "name" not in cmd or "cmd" not in cmd:
            raise ValueError(f"Command #{idx} must have 'name' and 'cmd' fields")

    print(f"[INFO] Loaded {len(data)} commands from {path}")
    return data

def run_noise_injection_and_algorithms(dataset_name: str,
                                       dataset_dir: str,
                                       neo4j_dir: str,
                                       version: str,
                                       node_properties,
                                       edge_properties,
                                       relationship_types,
                                       node_labels,
                                       commands):
    global OUTPUT_BASE_DIR, NOISE_LEVELS, LABEL_PERCENTS

    output_dir = os.path.join(OUTPUT_BASE_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for noise in NOISE_LEVELS:
        print("=" * 40)
        print(f"Dataset: {dataset_name} | Noise: {noise}%")
        print("=" * 40)

        # Restart DB from clean dump
        stop_neo4j(neo4j_dir)
        dump_file, _ = detect_dump_and_version(dataset_dir)
        load_dump(neo4j_dir, dump_file, version)
        start_neo4j_and_wait(neo4j_dir)

        # Save original labels
        cypher(
            neo4j_dir,
            "MATCH (n) SET n.original_label = labels(n)",
            log_path=os.path.join(output_dir, f"log_noise{noise}_init.txt")
        )

        frac = noise / 100.0

        for prop in node_properties:
            print(f"Removing node property {prop} with fraction {frac}")
            query = (
                f"MATCH (n) WHERE rand() < {frac} AND '{prop}' IN keys(n) "
                f"REMOVE n.`{prop}`"
            )
            cypher(
                neo4j_dir,
                query,
                log_path=os.path.join(output_dir, f"log_noise{noise}_node_{prop}.txt")
            )

        cypher(
            neo4j_dir,
            "MATCH (n) UNWIND keys(n) AS k "
            "RETURN k, count(*) AS cnt ORDER BY k LIMIT 10",
            log_path=os.path.join(output_dir, f"node_props_noise{noise}.txt")
        )

        for prop in edge_properties:
            print(f"Removing relationship property {prop} with fraction {frac}")
            query = (
                f"MATCH ()-[r]-() WHERE rand() < {frac} AND '{prop}' IN keys(r) "
                f"REMOVE r.`{prop}`"
            )
            cypher(
                neo4j_dir,
                query,
                log_path=os.path.join(output_dir, f"log_noise{noise}_rel_{prop}.txt")
            )

        # Debug relationships
        cypher(
            neo4j_dir,
            "MATCH ()-[r]-() UNWIND keys(r) AS k "
            "RETURN k, count(*) AS cnt ORDER BY k LIMIT 10",
            log_path=os.path.join(output_dir, f"rel_props_noise{noise}.txt")
        )

        cypher(
            neo4j_dir,
            "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt "
            "ORDER BY rel_type LIMIT 10",
            log_path=os.path.join(output_dir, f"rel_types_noise{noise}.txt")
        )

        for perc in LABEL_PERCENTS:
            print(f"Removing labels for fraction {perc}")
            if node_labels:
                labels_str = ":".join(node_labels)
                query = (
                    f"MATCH (n) WHERE rand() < {perc} "
                    f"REMOVE n:{labels_str}"
                )
                cypher(
                    neo4j_dir,
                    query,
                    log_path=os.path.join(output_dir, f"log_noise{noise}_labels{perc}.txt")
                )
            else:
                print(f"[INFO] No node_labels for dataset {dataset_name} – skipping label removal.")

            time.sleep(2)

            # Debug labels
            cypher(
                neo4j_dir,
                "MATCH (n) UNWIND labels(n) AS l "
                "RETURN l, count(*) AS cnt ORDER BY l",
                log_path=os.path.join(output_dir, f"labels_noise{noise}_perc{perc}.txt")
            )

            # === Run user-defined commands ===
            ds_upper = dataset_name.upper()

            for cmd_def in commands:
                name = cmd_def["name"]
                cmd = cmd_def["cmd"]
                cwd = cmd_def.get("cwd", None)

                log_filename = f"output_{ds_upper}_noise{noise}_labels{perc}_{name}.txt"
                log_path = os.path.join(output_dir, log_filename)

                print(f"Running command '{name}' for dataset={dataset_name}, "
                      f"noise={noise}, labels={perc}")
                run(cmd, cwd=cwd, log_file=log_path)
                clean_spark_tmp()

        stop_neo4j(neo4j_dir)

    print(f"All noise levels processed for dataset {dataset_name}.")



def load_config(config_path: str):
    global DATASETS_BASE_DIR, OUTPUT_BASE_DIR, COMMANDS_FILE
    global NEO4J_PASSWORD, NEO4J_PORT, NOISE_LEVELS, LABEL_PERCENTS, NEO4J_DIRS

    defaults = {
        "datasets_dir": "./datasets",
        "output_dir": "./output",
        "commands_file": "./benchmark_commands.json",
        "neo4j_password": "password",
        "neo4j_port": 7687,
        "noise_levels": [0, 10, 20, 30, 40],
        "label_percents": [0.0, 0.5, 1.0],
        "neo4j_dirs": {}
    }

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Create one based on the example."
        )

    with open(config_path) as f:
        cfg = json.load(f)

    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v

    DATASETS_BASE_DIR = cfg["datasets_dir"]
    OUTPUT_BASE_DIR = cfg["output_dir"]
    COMMANDS_FILE = cfg["commands_file"]
    NEO4J_PASSWORD = cfg["neo4j_password"]
    NEO4J_PORT = cfg["neo4j_port"]
    NOISE_LEVELS = cfg["noise_levels"]
    LABEL_PERCENTS = cfg["label_percents"]
    NEO4J_DIRS = cfg["neo4j_dirs"]

    print("[CONFIG] DATASETS_BASE_DIR:", DATASETS_BASE_DIR)
    print("[CONFIG] OUTPUT_BASE_DIR:", OUTPUT_BASE_DIR)
    print("[CONFIG] COMMANDS_FILE:", COMMANDS_FILE)
    print("[CONFIG] NEO4J_PORT:", NEO4J_PORT)
    print("[CONFIG] NOISE_LEVELS:", NOISE_LEVELS)
    print("[CONFIG] LABEL_PERCENTS:", LABEL_PERCENTS)
    print("[CONFIG] NEO4J_DIRS:", NEO4J_DIRS)


def main():
    parser = argparse.ArgumentParser(description="Property Graph Benchmark")
    parser.add_argument(
        "--config",
        "-c",
        default="benchmark_config.json",
        help="Path to benchmark_config.json (default: ./benchmark_config.json)"
    )
    args = parser.parse_args()

    load_config(args.config)

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    commands = load_commands(COMMANDS_FILE)

    for entry in sorted(os.listdir(DATASETS_BASE_DIR)):
        dataset_dir = os.path.join(DATASETS_BASE_DIR, entry)
        if not os.path.isdir(dataset_dir):
            continue

        dataset_name = entry
        print("\n" + "#" * 60)
        print(f"Processing dataset: {dataset_name}")
        print("#" * 60)

        try:
            dump_file, version = detect_dump_and_version(dataset_dir)
        except Exception as e:
            print(f"[ERROR] Skipping {dataset_name}: {e}")
            continue

        print(f"  Dump: {dump_file}")
        print(f"  Neo4j version detected: {version}")

        try:
            neo4j_dir = neo4j_dir_for_version(version)
        except Exception as e:
            print(f"[ERROR] Skipping {dataset_name}: {e}")
            continue

        node_properties = read_csv_list(os.path.join(dataset_dir, "node_properties.csv"))
        edge_properties = read_csv_list(os.path.join(dataset_dir, "edge_properties.csv"))
        relationship_types = read_csv_list(os.path.join(dataset_dir, "edge_labels.csv"))
        node_labels = read_csv_list(os.path.join(dataset_dir, "node_labels.csv"))

        try:
            run_noise_injection_and_algorithms(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                neo4j_dir=neo4j_dir,
                version=version,
                node_properties=node_properties,
                edge_properties=edge_properties,
                relationship_types=relationship_types,
                node_labels=node_labels,
                commands=commands
            )
        except Exception as e:
            print(f"[ERROR] Benchmark failed for dataset {dataset_name}: {e}")

    print("All datasets processed.")


if __name__ == "__main__":
    main()
