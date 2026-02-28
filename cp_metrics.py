#!/usr/bin/env python3

import argparse
import json
import os
import time
from datetime import datetime

from neo4j import GraphDatabase


# CP metrics + dataset summary counts for IYP.
Q_CP2_CP3_CP8 = """
CYPHER runtime=parallel
MATCH (n)
RETURN count(n) AS nodes,
       avg(size(labels(n))) AS cp3_avg_label_cardinality,
       toFloat(sum(CASE WHEN size(labels(n)) = 0 THEN 1 ELSE 0 END)) / count(n) AS cp2_unlabeled_ratio
"""

Q_CP1_CP5 = """
CYPHER runtime=parallel
MATCH (n)
WITH apoc.text.join(apoc.coll.sort(labels(n)), ":") AS nodeType,
     apoc.text.join(apoc.coll.sort(keys(n)), ":") AS propSet
WITH nodeType, propSet, count(*) AS node_count_for_pattern
WITH nodeType, count(*) AS patterns_per_node_type
RETURN avg(patterns_per_node_type) AS cp1_avg_node_patterns_per_node_type,
       sum(patterns_per_node_type) AS cp5_total_node_patterns,
       count(*) AS node_types
"""

Q_CP5_GLOBAL_PROPSETS = """
CYPHER runtime=parallel
MATCH (n)
WITH apoc.text.join(apoc.coll.sort(keys(n)), ":") AS propSet
RETURN count(DISTINCT propSet) AS cp5_global_distinct_property_sets
"""

Q_CP7_CP8 = """
CYPHER runtime=parallel
MATCH ()-[r]->()
RETURN count(r) AS edges
"""

Q_CP7 = """
CYPHER runtime=parallel
MATCH (s)-[r]->(t)
WITH type(r) AS relType,
     apoc.text.join(apoc.coll.sort(labels(s)), ":") AS sourceLabelCombo,
     apoc.text.join(apoc.coll.sort(labels(t)), ":") AS targetLabelCombo,
     apoc.text.join(apoc.coll.sort(keys(r)), ":") AS propSet
WITH relType, sourceLabelCombo, targetLabelCombo, propSet, count(*) AS edge_count_for_pattern
WITH relType, sourceLabelCombo, targetLabelCombo, count(*) AS patterns_per_edge_type
RETURN avg(patterns_per_edge_type) AS cp7_avg_edge_patterns_per_edge_type,
       sum(patterns_per_edge_type) AS edge_patterns,
       count(*) AS edge_types
"""

Q_REL_TYPES = """
CALL db.relationshipTypes() YIELD relationshipType
RETURN relationshipType
ORDER BY relationshipType
"""

Q_CP7_PER_RELTYPE = """
CYPHER runtime=parallel
MATCH (s)-[r]->(t)
WHERE type(r) = $rel_type
WITH apoc.text.join(apoc.coll.sort(labels(s)), ":") AS sourceLabelCombo,
     apoc.text.join(apoc.coll.sort(labels(t)), ":") AS targetLabelCombo,
     apoc.text.join(apoc.coll.sort(keys(r)), ":") AS propSet
WITH sourceLabelCombo, targetLabelCombo, propSet, count(*) AS edge_count_for_pattern
WITH sourceLabelCombo, targetLabelCombo, count(*) AS patterns_per_edge_type
RETURN sum(patterns_per_edge_type) AS edge_patterns,
       count(*) AS edge_types
"""

Q_NODE_LABELS = """
CYPHER runtime=parallel
MATCH (n)
UNWIND labels(n) AS label
RETURN count(DISTINCT label) AS node_labels
"""

Q_EDGE_LABELS = """
CYPHER runtime=parallel
MATCH ()-[r]->()
RETURN count(DISTINCT type(r)) AS edge_labels
"""

Q_TOTAL_PROPERTIES = """
CALL db.propertyKeys() YIELD propertyKey
RETURN count(*) AS total_properties
"""


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_single_row_with_retries(session, query: str, name: str, retries: int, sleep_sec: int, params=None):
    attempt = 1
    while True:
        try:
            print(f"[{ts()}] Running {name} (attempt {attempt})...")
            row = session.run(query, params or {}).single()
            if row is None:
                raise RuntimeError(f"{name} returned no rows.")
            print(f"[{ts()}] Finished {name}.")
            return dict(row)
        except Exception as exc:
            if attempt > retries:
                raise RuntimeError(f"{name} failed after {attempt} attempts: {exc}") from exc
            print(f"[{ts()}] {name} failed ({exc}). Retrying in {sleep_sec}s...")
            time.sleep(sleep_sec)
            attempt += 1


def main():
    parser = argparse.ArgumentParser(description="Compute CP metrics and key dataset counts.")
    parser.add_argument(
        "--dataset",
        default="iyp",
        help="Dataset name used in the output metadata and default output path.",
    )
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--db", default="neo4j")
    parser.add_argument("--out")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=int, default=10)
    parser.add_argument(
        "--skip-cp7",
        action="store_true",
        help="Skip CP7 (heaviest query over all relationships).",
    )
    parser.add_argument(
        "--cp7-mode",
        choices=["per-reltype", "single-query"],
        default="per-reltype",
        help="How to compute CP7. per-reltype is usually more stable on very large graphs.",
    )
    args = parser.parse_args()

    if not args.out:
        args.out = os.path.join(f"stats_{args.dataset}", "cp_metrics.json")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session(database=args.db) as session:
            cp2_cp3_cp8 = run_single_row_with_retries(
                session, Q_CP2_CP3_CP8, "CP2/CP3/Nodes", args.retries, args.retry_sleep
            )
            cp1_cp5 = run_single_row_with_retries(
                session, Q_CP1_CP5, "CP1/CP5", args.retries, args.retry_sleep
            )
            cp5_global = run_single_row_with_retries(
                session, Q_CP5_GLOBAL_PROPSETS, "CP5(global prop sets)", args.retries, args.retry_sleep
            )
            edge_count = run_single_row_with_retries(
                session, Q_CP7_CP8, "Edges", args.retries, args.retry_sleep
            )
            node_labels = run_single_row_with_retries(
                session, Q_NODE_LABELS, "Node labels", args.retries, args.retry_sleep
            )
            edge_labels = run_single_row_with_retries(
                session, Q_EDGE_LABELS, "Edge labels", args.retries, args.retry_sleep
            )
            total_properties = run_single_row_with_retries(
                session, Q_TOTAL_PROPERTIES, "Total properties", args.retries, args.retry_sleep
            )

            cp7_data = {}
            if args.skip_cp7:
                cp7_data = {
                    "cp7_avg_edge_patterns_per_edge_type": None,
                    "edge_patterns": None,
                    "edge_types": None,
                    "cp7_skipped": True,
                    "cp7_mode": "skipped",
                }
            else:
                if args.cp7_mode == "single-query":
                    cp7_data = run_single_row_with_retries(
                        session, Q_CP7, "CP7(single-query)", args.retries, args.retry_sleep
                    )
                    cp7_data["cp7_mode"] = "single-query"
                else:
                    rel_types_rows = session.run(Q_REL_TYPES)
                    rel_types = [r["relationshipType"] for r in rel_types_rows]
                    total_edge_patterns = 0
                    total_edge_types = 0
                    for rel_type in rel_types:
                        per_type = run_single_row_with_retries(
                            session,
                            Q_CP7_PER_RELTYPE,
                            f"CP7(relType={rel_type})",
                            args.retries,
                            args.retry_sleep,
                            params={"rel_type": rel_type},
                        )
                        total_edge_patterns += int(per_type["edge_patterns"] or 0)
                        total_edge_types += int(per_type["edge_types"] or 0)

                    cp7_data = {
                        "edge_patterns": total_edge_patterns,
                        "edge_types": total_edge_types,
                        "cp7_avg_edge_patterns_per_edge_type": (
                            float(total_edge_patterns) / total_edge_types if total_edge_types else None
                        ),
                        "cp7_mode": "per-reltype",
                    }
                cp7_data["cp7_skipped"] = False

        nodes = int(cp2_cp3_cp8["nodes"])
        edges = int(edge_count["edges"])
        cp2_ratio = float(cp2_cp3_cp8["cp2_unlabeled_ratio"])
        cp3 = float(cp2_cp3_cp8["cp3_avg_label_cardinality"])
        cp1 = float(cp1_cp5["cp1_avg_node_patterns_per_node_type"])
        cp5 = int(cp1_cp5["cp5_total_node_patterns"])
        cp5_global_distinct_propsets = int(cp5_global["cp5_global_distinct_property_sets"])
        node_types = int(cp1_cp5["node_types"])

        result = {
            "dataset": args.dataset,
            "nodes": nodes,
            "edges": edges,
            "node_types": node_types,
            "edge_types": cp7_data.get("edge_types"),
            "node_labels": int(node_labels["node_labels"]),
            "edge_labels": int(edge_labels["edge_labels"]),
            "node_patterns": cp5,
            "edge_patterns": cp7_data.get("edge_patterns"),
            "total_properties": int(total_properties["total_properties"]),
            "cp8_nodes_plus_edges": nodes + edges,
            "cp1_avg_node_patterns_per_node_type": cp1,
            "cp2_unlabeled_nodes_percent": cp2_ratio * 100.0,
            "cp3_avg_label_cardinality_per_node": cp3,
            "cp5_total_node_patterns": cp5,
            "cp5_global_distinct_property_sets": cp5_global_distinct_propsets,
            "cp7_avg_edge_patterns_per_edge_type": cp7_data.get("cp7_avg_edge_patterns_per_edge_type"),
            "cp7_skipped": cp7_data.get("cp7_skipped", False),
            "cp7_mode": cp7_data.get("cp7_mode"),
            "computed_at": ts(),
        }

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[{ts()}] Wrote: {args.out}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
