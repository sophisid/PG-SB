import os
import re
import glob
import time
import tempfile
import subprocess
import csv
import json
import argparse
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from urllib.parse import urlparse

from evaluation import evaluate_run_from_csvs

DATASETS_BASE_DIR = None
OUTPUT_BASE_DIR = None
COMMANDS_FILE = None
NEO4J_PASSWORD = None
NEO4J_PORT = None
NOISE_LEVELS = None
LABEL_PERCENTS = None
NEO4J_DIRS = None
NEO4J_MODE = None
NEO4J_DESKTOP_DIRS = None
NEO4J_AUTO_DOWNLOAD = None
NEO4J_DOWNLOAD_DIR = None
NEO4J_MEMORY = None
DATASET_ORDER = None
RUN_EXTERNAL_COMMANDS = None
QUERY_BATCH_SIZE = None
EVAL_ENABLED = False
PREPARED_COMMAND_ENVS = set()


def run(cmd, cwd=None, log_file=None, check=True):
    print(f"[RUN] {cmd} (cwd={cwd})")
    if log_file:
        with open(log_file, "a") as f:
            proc = subprocess.run(cmd, cwd=cwd, shell=True, stdout=f, stderr=subprocess.STDOUT)
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

    m = re.search(r"neo4j-(\d+\.\d+\.\d+)", base)
    if not m:
        raise ValueError(
            f"Could not detect Neo4j version from dump name: {dump_file}. "
            f"Expected something like '*neo4j-X.Y.Z.dump'."
        )

    return dump_file, m.group(1).strip()


def neo4j_dir_for_version(version: str) -> str:
    global NEO4J_MODE, NEO4J_DIRS, NEO4J_DESKTOP_DIRS, NEO4J_AUTO_DOWNLOAD, NEO4J_DOWNLOAD_DIR

    if NEO4J_MODE == "desktop":
        if version not in NEO4J_DESKTOP_DIRS:
            raise FileNotFoundError(
                f"No Neo4j Desktop DBMS directory configured for version {version}. "
                f"Add it under 'neo4j_desktop_dirs' in your config file."
            )
        neo4j_dir = os.path.abspath(os.path.expanduser(NEO4J_DESKTOP_DIRS[version]))
        validate_neo4j_home(neo4j_dir, version, NEO4J_MODE)
        return neo4j_dir

    configured_dir = NEO4J_DIRS.get(version)
    if configured_dir:
        neo4j_dir = os.path.abspath(os.path.expanduser(configured_dir))
    else:
        neo4j_dir = os.path.abspath(
            os.path.join(os.path.expanduser(NEO4J_DOWNLOAD_DIR), f"neo4j-community-{version}")
        )

    if not os.path.isdir(neo4j_dir):
        if not NEO4J_AUTO_DOWNLOAD:
            raise FileNotFoundError(
                f"Neo4j Community {version} is not available at {neo4j_dir}. "
                f"Either install it there or enable 'neo4j_auto_download'."
            )
        download_neo4j_community(version, neo4j_dir)

    validate_neo4j_home(neo4j_dir, version, NEO4J_MODE)
    configure_neo4j_memory(neo4j_dir, version)
    ensure_initial_password(neo4j_dir, version)
    return neo4j_dir


def validate_neo4j_home(neo4j_dir: str, version: str, mode: str):
    required = [
        os.path.join(neo4j_dir, "bin", "neo4j"),
        os.path.join(neo4j_dir, "bin", "neo4j-admin"),
        os.path.join(neo4j_dir, "bin", "cypher-shell"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"The configured Neo4j {mode} directory for version {version} is not a valid DBMS home: {neo4j_dir}. "
            f"Missing: {', '.join(missing)}"
        )


def community_archive_candidates(version: str):
    if sys.platform.startswith("win"):
        filename = f"neo4j-community-{version}-windows.zip"
    else:
        filename = f"neo4j-community-{version}-unix.tar.gz"
    return [
        (f"https://dist.neo4j.org/{filename}", filename),
        (f"https://neo4j.com/artifact.php?name={filename}", filename),
    ]


def extract_archive(archive_path: str, destination_dir: str):
    if archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(destination_dir)
        return
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(destination_dir)
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def download_neo4j_community(version: str, target_dir: str):
    parent_dir = os.path.dirname(target_dir.rstrip(os.sep)) or "."
    os.makedirs(parent_dir, exist_ok=True)

    print(f"[INFO] Neo4j Community {version} not found locally. Downloading...")

    with tempfile.TemporaryDirectory(prefix=f"neo4j-community-{version}-") as tmp_dir:
        archive_path = None
        last_error = None

        for url, filename in community_archive_candidates(version):
            candidate_archive = os.path.join(tmp_dir, filename)
            try:
                print(f"[DOWNLOAD] {url}")
                urllib.request.urlretrieve(url, candidate_archive)
                archive_path = candidate_archive
                break
            except Exception as exc:
                last_error = exc
                print(f"[WARN] Download failed from {url}: {exc}")

        if archive_path is None:
            raise RuntimeError(
                f"Failed to download Neo4j Community {version}. "
                f"Tried the official distribution URLs. Last error: {last_error}"
            )

        extract_root = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_root, exist_ok=True)
        extract_archive(archive_path, extract_root)

        extracted_dirs = [
            os.path.join(extract_root, entry)
            for entry in os.listdir(extract_root)
            if os.path.isdir(os.path.join(extract_root, entry))
        ]
        if len(extracted_dirs) != 1:
            raise RuntimeError(
                f"Unexpected archive layout while extracting Neo4j Community {version}: {extracted_dirs}"
            )

        extracted_dir = extracted_dirs[0]
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.move(extracted_dir, target_dir)

    print(f"[INFO] Neo4j Community {version} installed at {target_dir}")


def ensure_initial_password(neo4j_dir: str, version: str):
    global NEO4J_PASSWORD

    auth_candidates = [
        os.path.join(neo4j_dir, "data", "dbms", "auth"),
        os.path.join(neo4j_dir, "data", "dbms", "auth.ini"),
    ]
    if any(os.path.exists(path) for path in auth_candidates):
        return

    print("[INFO] Setting initial Neo4j password for a fresh Community installation.")
    if version.startswith("4."):
        cmd = f'{neo4j_dir}/bin/neo4j-admin set-initial-password "{NEO4J_PASSWORD}"'
    else:
        cmd = f'{neo4j_dir}/bin/neo4j-admin dbms set-initial-password "{NEO4J_PASSWORD}"'
    run(cmd, check=False)


def upsert_neo4j_conf_setting(conf_path: str, key: str, value: str):
    lines = []
    if os.path.exists(conf_path):
        with open(conf_path, encoding="utf-8") as f:
            lines = f.readlines()

    target_line = f"{key}={value}\n"
    replaced = False
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped == key or stripped.startswith(f"{key}=") or stripped.startswith(f"#{key}="):
            if not replaced:
                new_lines.append(target_line)
                replaced = True
            continue
        new_lines.append(line)

    if not replaced:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        new_lines.append(target_line)

    with open(conf_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def configure_neo4j_memory(neo4j_dir: str, version: str):
    global NEO4J_MEMORY

    conf_path = os.path.join(neo4j_dir, "conf", "neo4j.conf")
    memory_cfg = NEO4J_MEMORY or {}
    heap_initial = memory_cfg.get("heap_initial", "8G")
    heap_max = memory_cfg.get("heap_max", "12G")
    pagecache = memory_cfg.get("pagecache", "8G")

    if version.startswith("4."):
        settings = {
            "dbms.memory.heap.initial_size": heap_initial,
            "dbms.memory.heap.max_size": heap_max,
            "dbms.memory.pagecache.size": pagecache,
        }
    else:
        settings = {
            "server.memory.heap.initial_size": heap_initial,
            "server.memory.heap.max_size": heap_max,
            "server.memory.pagecache.size": pagecache,
        }

    for key, value in settings.items():
        upsert_neo4j_conf_setting(conf_path, key, value)

    print(f"[INFO] Applied Neo4j memory settings in {conf_path}: {settings}")


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

    print(f"Killing any process on port {NEO4J_PORT}")
    subprocess.run(
        f'pids=$(lsof -t -i :{NEO4J_PORT}); if [ -n "$pids" ]; then kill -9 $pids; fi',
        shell=True
    )

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
    run(f"{neo4j_dir}/bin/neo4j start")
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
        time.sleep(10)


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


def batched_set_original_labels_query(batch_size: int) -> str:
    return f"""
CALL {{
  MATCH (n)
  WITH n
  SET n.original_label = labels(n)
}} IN TRANSACTIONS OF {batch_size} ROWS
"""


def batched_remove_node_property_query(prop: str, fraction: float, batch_size: int) -> str:
    return f"""
CALL {{
  MATCH (n)
  WHERE rand() < {fraction} AND '{prop}' IN keys(n)
  WITH n
  REMOVE n.`{prop}`
}} IN TRANSACTIONS OF {batch_size} ROWS
"""


def batched_remove_edge_property_query(prop: str, fraction: float, batch_size: int) -> str:
    return f"""
CALL {{
  MATCH ()-[r]-()
  WHERE rand() < {fraction} AND '{prop}' IN keys(r)
  WITH r
  REMOVE r.`{prop}`
}} IN TRANSACTIONS OF {batch_size} ROWS
"""


def batched_remove_labels_query(node_labels, fraction: float, batch_size: int) -> str:
    labels_str = ":".join(node_labels)
    return f"""
CALL {{
  MATCH (n)
  WHERE rand() < {fraction}
  WITH n
  REMOVE n:{labels_str}
}} IN TRANSACTIONS OF {batch_size} ROWS
"""


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
            if first.lower() in {"property", "properties", "label", "labels", "name", "relationship", "type"}:
                continue
            values.append(first)
    print(f"[INFO] Loaded {len(values)} items from {csv_path}")
    return values


def load_commands(path: str):
    if not os.path.exists(path):
        print(f"[INFO] Commands file not found ({path}), no external commands will be run.")
        return []

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Commands file {path} must contain a JSON list")

    for idx, cmd in enumerate(data):
        if "name" not in cmd or "cmd" not in cmd:
            raise ValueError(f"Command #{idx} must have 'name' and 'cmd' fields")

    print(f"[INFO] Loaded {len(data)} commands from {path}")
    return data


def default_clone_dir_for_repo(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    repo_name = os.path.basename(parsed.path.rstrip("/"))
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    return os.path.abspath(repo_name or "repo")


def ensure_repo_checkout(cmd_def: dict):
    repo_url = cmd_def.get("repo")
    if not repo_url:
        return

    clone_dir = os.path.abspath(os.path.expanduser(
        cmd_def.get("clone_dir") or default_clone_dir_for_repo(repo_url)
    ))
    branch = cmd_def.get("branch")
    update_existing = cmd_def.get("update_existing", True)

    if not os.path.isdir(clone_dir):
        parent_dir = os.path.dirname(clone_dir) or "."
        os.makedirs(parent_dir, exist_ok=True)
        clone_cmd = f'git clone "{repo_url}" "{clone_dir}"'
        if branch:
            clone_cmd = f'git clone --branch "{branch}" "{repo_url}" "{clone_dir}"'
        print(f"[INFO] Cloning repository for command '{cmd_def['name']}' into {clone_dir}")
        run(clone_cmd)
    elif update_existing:
        print(f"[INFO] Updating repository for command '{cmd_def['name']}' in {clone_dir}")
        run("git fetch --all --tags", cwd=clone_dir)
        if branch:
            run(f'git checkout "{branch}"', cwd=clone_dir)
            run(f'git pull --ff-only origin "{branch}"', cwd=clone_dir)
        else:
            run("git pull --ff-only", cwd=clone_dir)

    if not cmd_def.get("cwd"):
        cmd_def["cwd"] = clone_dir


def command_identity(cmd_def: dict):
    repo = cmd_def.get("repo", "")
    cwd = os.path.abspath(os.path.expanduser(cmd_def.get("cwd") or ""))
    setup_cmd = cmd_def.get("setup_cmd", "")
    return repo, cwd, setup_cmd


def ensure_command_dependencies(cmd_def: dict):
    setup_cmd = cmd_def.get("setup_cmd")
    if not setup_cmd:
        return

    identity = command_identity(cmd_def)
    if identity in PREPARED_COMMAND_ENVS:
        return

    cwd = cmd_def.get("cwd", None)
    print(f"[INFO] Installing dependencies for command '{cmd_def['name']}'")
    run(setup_cmd, cwd=cwd)
    PREPARED_COMMAND_ENVS.add(identity)


def run_noise_injection_and_algorithms(
    dataset_name: str,
    dataset_dir: str,
    neo4j_dir: str,
    version: str,
    node_properties,
    edge_properties,
    relationship_types,
    node_labels,
    commands
):
    global OUTPUT_BASE_DIR, NOISE_LEVELS, LABEL_PERCENTS, EVAL_ENABLED, QUERY_BATCH_SIZE

    output_dir = os.path.join(OUTPUT_BASE_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for noise in NOISE_LEVELS:
        print("=" * 40)
        print(f"Dataset: {dataset_name} | Noise: {noise}%")
        print("=" * 40)

        stop_neo4j(neo4j_dir)
        dump_file, _ = detect_dump_and_version(dataset_dir)
        load_dump(neo4j_dir, dump_file, version)
        start_neo4j_and_wait(neo4j_dir)

        # Save original labels
        print(f"[STEP] Saving original labels in batches of {QUERY_BATCH_SIZE}")
        cypher(
            neo4j_dir,
            batched_set_original_labels_query(QUERY_BATCH_SIZE),
            log_path=os.path.join(output_dir, f"log_noise{noise}_init.txt")
        )

        frac = noise / 100.0

        for prop in node_properties:
            print(f"[STEP] Removing node property '{prop}' in batches of {QUERY_BATCH_SIZE}")
            query = batched_remove_node_property_query(prop, frac, QUERY_BATCH_SIZE)
            cypher(
                neo4j_dir,
                query,
                log_path=os.path.join(output_dir, f"log_noise{noise}_node_{prop}.txt")
            )

        cypher(
            neo4j_dir,
            "MATCH (n) UNWIND keys(n) AS k RETURN k, count(*) AS cnt ORDER BY k LIMIT 10",
            log_path=os.path.join(output_dir, f"node_props_noise{noise}.txt")
        )

        for prop in edge_properties:
            print(f"[STEP] Removing edge property '{prop}' in batches of {QUERY_BATCH_SIZE}")
            query = batched_remove_edge_property_query(prop, frac, QUERY_BATCH_SIZE)
            cypher(
                neo4j_dir,
                query,
                log_path=os.path.join(output_dir, f"log_noise{noise}_rel_{prop}.txt")
            )

        cypher(
            neo4j_dir,
            "MATCH ()-[r]-() UNWIND keys(r) AS k RETURN k, count(*) AS cnt ORDER BY k LIMIT 10",
            log_path=os.path.join(output_dir, f"rel_props_noise{noise}.txt")
        )

        cypher(
            neo4j_dir,
            "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt ORDER BY rel_type LIMIT 10",
            log_path=os.path.join(output_dir, f"rel_types_noise{noise}.txt")
        )

        for perc in LABEL_PERCENTS:
            print(f"Removing labels for fraction {perc}")
            if node_labels:
                print(f"[STEP] Removing labels in batches of {QUERY_BATCH_SIZE}")
                query = batched_remove_labels_query(node_labels, perc, QUERY_BATCH_SIZE)
                cypher(
                    neo4j_dir,
                    query,
                    log_path=os.path.join(output_dir, f"log_noise{noise}_labels{perc}.txt")
                )
            else:
                print(f"[INFO] No node_labels for dataset {dataset_name}")

            time.sleep(2)

            cypher(
                neo4j_dir,
                "MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS cnt ORDER BY l",
                log_path=os.path.join(output_dir, f"labels_noise{noise}_perc{perc}.txt")
            )

            ds_upper = dataset_name.upper()

            for cmd_def in commands:
                name = cmd_def["name"]
                cmd = cmd_def["cmd"]
                cwd = cmd_def.get("cwd", None)

                log_filename = f"output_{ds_upper}_noise{noise}_labels{perc}_{name}.txt"
                log_path = os.path.join(output_dir, log_filename)

                if RUN_EXTERNAL_COMMANDS:
                    print(f"Running command '{name}'")
                    run(cmd, cwd=cwd, log_file=log_path)
                    clean_spark_tmp()
                else:
                    print(f"[INFO] Skipping external command '{name}' because run_external_commands=false")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write("Skipped by benchmark configuration: run_external_commands=false\n")

                if not EVAL_ENABLED:
                    print("[EVAL] Disabled – skipping.")
                    continue

                label_str = f"{perc:.2f}".replace(",", ".")
                base = f"{ds_upper}_noise{noise}_labels{label_str}_{name.upper()}"

                original_nodes_csv = os.path.join(output_dir, f"original_nodes_{base}.csv")
                predicted_nodes_csv = os.path.join(output_dir, f"predicted_nodes_{base}.csv")
                original_edges_csv = os.path.join(output_dir, f"original_edges_{base}.csv")
                predicted_edges_csv = os.path.join(output_dir, f"predicted_edges_{base}.csv")

                required = [
                    original_nodes_csv,
                    predicted_nodes_csv,
                    original_edges_csv,
                    predicted_edges_csv
                ]

                if all(os.path.exists(p) for p in required):
                    print(
                        f"[EVAL] Running evaluation for {dataset_name}, "
                        f"noise={noise}, labels={perc}, method={name}"
                    )
                    evaluate_run_from_csvs(
                        output_dir=output_dir,
                        dataset=dataset_name,
                        method=name,
                        noise=noise,
                        label_percent=perc,
                        original_nodes_csv=original_nodes_csv,
                        predicted_nodes_csv=predicted_nodes_csv,
                        original_edges_csv=original_edges_csv,
                        predicted_edges_csv=predicted_edges_csv
                    )
                else:
                    print(f"[EVAL] Missing CSVs → skipping evaluation for method {name}")

        stop_neo4j(neo4j_dir)

    print(f"All noise levels processed for dataset {dataset_name}.")



def load_config(config_path: str):
    global DATASETS_BASE_DIR, OUTPUT_BASE_DIR, COMMANDS_FILE
    global NEO4J_PASSWORD, NEO4J_PORT, NOISE_LEVELS, LABEL_PERCENTS
    global NEO4J_DIRS, NEO4J_MODE, NEO4J_DESKTOP_DIRS, NEO4J_AUTO_DOWNLOAD, NEO4J_DOWNLOAD_DIR, NEO4J_MEMORY
    global DATASET_ORDER
    global RUN_EXTERNAL_COMMANDS
    global QUERY_BATCH_SIZE

    defaults = {
        "datasets_dir": "./datasets",
        "output_dir": "./output",
        "commands_file": "./benchmark_commands.json",
        "neo4j_password": "password",
        "neo4j_port": 7687,
        "noise_levels": [0, 10, 20, 30, 40],
        "label_percents": [0.0, 0.5, 1.0],
        "neo4j_mode": "community",
        "neo4j_dirs": {},
        "neo4j_desktop_dirs": {},
        "neo4j_auto_download": True,
        "neo4j_download_dir": "./neo4j_runtimes",
        "neo4j_memory": {
            "heap_initial": "8G",
            "heap_max": "12G",
            "pagecache": "8G"
        },
        "dataset_order": [],
        "run_external_commands": True,
        "query_batch_size": 10000
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
    NEO4J_MODE = cfg["neo4j_mode"]
    NEO4J_DIRS = cfg["neo4j_dirs"]
    NEO4J_DESKTOP_DIRS = cfg["neo4j_desktop_dirs"]
    NEO4J_AUTO_DOWNLOAD = cfg["neo4j_auto_download"]
    NEO4J_DOWNLOAD_DIR = cfg["neo4j_download_dir"]
    NEO4J_MEMORY = cfg["neo4j_memory"]
    DATASET_ORDER = cfg["dataset_order"]
    RUN_EXTERNAL_COMMANDS = cfg["run_external_commands"]
    QUERY_BATCH_SIZE = cfg["query_batch_size"]

    if NEO4J_MODE not in {"community", "desktop"}:
        raise ValueError("neo4j_mode must be either 'community' or 'desktop'")

    print("[CONFIG] DATASETS_BASE_DIR:", DATASETS_BASE_DIR)
    print("[CONFIG] OUTPUT_BASE_DIR:", OUTPUT_BASE_DIR)
    print("[CONFIG] COMMANDS_FILE:", COMMANDS_FILE)
    print("[CONFIG] NEO4J_MODE:", NEO4J_MODE)
    print("[CONFIG] NEO4J_PORT:", NEO4J_PORT)
    print("[CONFIG] NOISE_LEVELS:", NOISE_LEVELS)
    print("[CONFIG] LABEL_PERCENTS:", LABEL_PERCENTS)
    print("[CONFIG] NEO4J_DIRS:", NEO4J_DIRS)
    print("[CONFIG] NEO4J_DESKTOP_DIRS:", NEO4J_DESKTOP_DIRS)
    print("[CONFIG] NEO4J_AUTO_DOWNLOAD:", NEO4J_AUTO_DOWNLOAD)
    print("[CONFIG] NEO4J_DOWNLOAD_DIR:", NEO4J_DOWNLOAD_DIR)
    print("[CONFIG] NEO4J_MEMORY:", NEO4J_MEMORY)
    print("[CONFIG] DATASET_ORDER:", DATASET_ORDER)
    print("[CONFIG] RUN_EXTERNAL_COMMANDS:", RUN_EXTERNAL_COMMANDS)
    print("[CONFIG] QUERY_BATCH_SIZE:", QUERY_BATCH_SIZE)


def ordered_dataset_entries(base_dir: str):
    global DATASET_ORDER

    entries = [
        entry for entry in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, entry))
    ]
    entries.sort()

    if not DATASET_ORDER:
        return entries

    alias_map = {
        "starwars": "star-wars",
        "het": "hetio",
        "cord": "cord19",
        "fib": "fib25",
    }

    normalized_requested = []
    seen = set()
    for item in DATASET_ORDER:
        canonical = alias_map.get(item, item)
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized_requested.append(canonical)

    requested_existing = [entry for entry in normalized_requested if entry in entries]
    remaining = [entry for entry in entries if entry not in requested_existing]
    return requested_existing + remaining


def prepare_external_commands(commands):
    global RUN_EXTERNAL_COMMANDS

    if not RUN_EXTERNAL_COMMANDS:
        print("[INFO] External commands are disabled. Skipping repo checkout/setup.")
        return

    print("[INFO] Preparing external repositories and dependencies before dataset runs...")
    for cmd_def in commands:
        ensure_repo_checkout(cmd_def)
        ensure_command_dependencies(cmd_def)
    print("[INFO] External repositories are ready.")


def main():
    global EVAL_ENABLED
    parser = argparse.ArgumentParser(description="Property Graph Benchmark")
    parser.add_argument("--config", "-c", default="benchmark_config.json")
    parser.add_argument("--eval", action="store_true", help="Run evaluation if CSVs exist")
    args = parser.parse_args()
    EVAL_ENABLED = args.eval

    load_config(args.config)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    commands = load_commands(COMMANDS_FILE)
    prepare_external_commands(commands)

    for entry in ordered_dataset_entries(DATASETS_BASE_DIR):
        dataset_dir = os.path.join(DATASETS_BASE_DIR, entry)
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
