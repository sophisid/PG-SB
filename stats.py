#!/usr/bin/env python3

import argparse
import csv
import json
import os
from neo4j import GraphDatabase


# -------------------------
# BASIC COUNTS
# -------------------------

CYPHER_TOTAL_NODES = "MATCH (n) RETURN count(n) AS total_nodes"
CYPHER_TOTAL_EDGES = "MATCH ()-[r]->() RETURN count(r) AS total_edges"

# -------------------------
# NODE LABELS
# -------------------------

CYPHER_NODE_LABELS = """
CALL db.labels() YIELD label
RETURN label
ORDER BY label
"""

CYPHER_NODE_LABEL_COUNTS = """
MATCH (n)
UNWIND labels(n) AS label
RETURN label, count(*) AS count
ORDER BY count DESC, label
"""

# -------------------------
# EDGE LABELS (REL TYPES)
# -------------------------

CYPHER_REL_TYPES = """
CALL db.relationshipTypes() YIELD relationshipType
RETURN relationshipType
ORDER BY relationshipType
"""

CYPHER_REL_TYPE_COUNTS = """
MATCH ()-[r]->()
RETURN type(r) AS relType, count(*) AS count
ORDER BY count DESC, relType
"""

# -------------------------
# PROPERTY KEYS (global)
# -------------------------

CYPHER_NODE_PROPERTY_KEYS = """
MATCH (n)
UNWIND keys(n) AS key
RETURN DISTINCT key AS key
ORDER BY key
"""

CYPHER_EDGE_PROPERTY_KEYS = """
MATCH ()-[r]->()
UNWIND keys(r) AS key
RETURN DISTINCT key AS key
ORDER BY key
"""

# -------------------------
# YOUR TYPE DEFINITIONS
# -------------------------

CYPHER_NODE_TYPES = """
CALL apoc.meta.nodeTypeProperties()
YIELD nodeLabels, propertyName
WITH apoc.coll.sort(nodeLabels) AS labels, propertyName
WITH apoc.text.join(labels, ":") AS combinedLabels, propertyName
WITH combinedLabels, collect(distinct propertyName) AS properties
RETURN combinedLabels AS nodeType, properties
ORDER BY nodeType
"""

CYPHER_NODE_TYPE_COUNTS = """
MATCH (n)
WITH apoc.text.join(apoc.coll.sort(labels(n)), ":") AS nodeType
RETURN nodeType, count(*) AS count
ORDER BY count DESC, nodeType
"""

CYPHER_NODE_PATTERNS = """
MATCH (n)
WITH apoc.text.join(apoc.coll.sort(labels(n)), ":") AS nodeType,
     apoc.text.join(apoc.coll.sort(keys(n)), ":") AS propSet
RETURN nodeType, propSet, count(*) AS count
ORDER BY count DESC, nodeType, propSet
"""

CYPHER_EDGE_TYPES = """
MATCH (s)-[r]->(t)
WITH type(r) AS relType,
     apoc.text.join(apoc.coll.sort(labels(s)), ":") AS sourceLabelCombo,
     apoc.text.join(apoc.coll.sort(labels(t)), ":") AS targetLabelCombo,
     keys(r) AS props
WITH relType,
     collect(distinct sourceLabelCombo) AS sources,
     collect(distinct targetLabelCombo) AS targets,
     collect(distinct props) AS propsList
RETURN
  relType,
  sources,
  targets,
  apoc.coll.toSet(apoc.coll.flatten(propsList)) AS properties
ORDER BY relType
"""

CYPHER_EDGE_PATTERNS = """
MATCH (s)-[r]->(t)
WITH type(r) AS relType,
     apoc.text.join(apoc.coll.sort(labels(s)), ":") AS sourceLabelCombo,
     apoc.text.join(apoc.coll.sort(labels(t)), ":") AS targetLabelCombo,
     apoc.text.join(apoc.coll.sort(keys(r)), ":") AS propSet
RETURN relType, sourceLabelCombo, targetLabelCombo, propSet, count(*) AS count
ORDER BY count DESC, relType, sourceLabelCombo, targetLabelCombo, propSet
"""


# -------------------------
# UTILS
# -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: str, rows, fields) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_single_row_csv(path: str, row: dict) -> None:
    fields = list(row.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def pipe_join(values) -> str:
    if values is None:
        return ""
    # ensure unique + stable order
    uniq = sorted(set(str(v) for v in values))
    return "|".join(uniq)


def aggregate_edge_types(edge_types, rel_type_counts):
    counts_by_rel_type = {row["relType"]: row["count"] for row in rel_type_counts}
    aggregated = []

    for edge_type in edge_types:
        rel_type = edge_type["relType"]
        aggregated.append({
            "relType": rel_type,
            "sourceLabelCombo": pipe_join(edge_type.get("sources") or []),
            "targetLabelCombo": pipe_join(edge_type.get("targets") or []),
            "properties": pipe_join(edge_type.get("properties") or []),
            "count": int(counts_by_rel_type.get(rel_type, 0)),
        })

    aggregated.sort(key=lambda row: (-row["count"], row["relType"]))
    return aggregated


# -------------------------
# MAIN
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--db", default="neo4j")
    parser.add_argument("--out", default="stats")
    args = parser.parse_args()

    ensure_dir(args.out)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    with driver.session(database=args.db) as session:

        # Global totals
        total_nodes = int(session.run(CYPHER_TOTAL_NODES).single()["total_nodes"])
        total_edges = int(session.run(CYPHER_TOTAL_EDGES).single()["total_edges"])

        # Node labels (+ counts)
        node_labels = [r["label"] for r in session.run(CYPHER_NODE_LABELS)]
        node_label_counts = [{"label": r["label"], "count": int(r["count"])}
                             for r in session.run(CYPHER_NODE_LABEL_COUNTS)]

        # Edge labels == relationship types (+ counts)
        rel_types = [r["relationshipType"] for r in session.run(CYPHER_REL_TYPES)]
        rel_type_counts = [{"relType": r["relType"], "count": int(r["count"])}
                           for r in session.run(CYPHER_REL_TYPE_COUNTS)]

        # Property keys (global)
        node_property_keys = [r["key"] for r in session.run(CYPHER_NODE_PROPERTY_KEYS)]
        edge_property_keys = [r["key"] for r in session.run(CYPHER_EDGE_PROPERTY_KEYS)]
        all_property_keys = sorted(set(node_property_keys).union(edge_property_keys))

        # Node types (+ props) + counts
        node_types = [{"nodeType": r["nodeType"], "properties": r["properties"]}
                      for r in session.run(CYPHER_NODE_TYPES)]
        node_type_counts = [{"nodeType": r["nodeType"], "count": int(r["count"])}
                            for r in session.run(CYPHER_NODE_TYPE_COUNTS)]
        node_patterns = [{"nodeType": r["nodeType"], "propSet": r["propSet"], "count": int(r["count"])}
                         for r in session.run(CYPHER_NODE_PATTERNS)]

        # Edge types / counts / patterns
        edge_types = [{"relType": r["relType"], "sources": r["sources"], "targets": r["targets"], "properties": r["properties"]}
                      for r in session.run(CYPHER_EDGE_TYPES)]
        edge_patterns = [{"relType": r["relType"],
                          "sourceLabelCombo": r["sourceLabelCombo"],
                          "targetLabelCombo": r["targetLabelCombo"],
                          "propSet": r["propSet"],
                          "count": int(r["count"])}
                         for r in session.run(CYPHER_EDGE_PATTERNS)]

    driver.close()

    # -------------------------
    # NEW: nodeType + properties (no duplicates) + count
    # -------------------------
    counts_by_type = {x["nodeType"]: x["count"] for x in node_type_counts}

    node_types_properties_counts = []
    for nt in node_types:
        t = nt["nodeType"]
        props = nt.get("properties") or []
        node_types_properties_counts.append({
            "nodeType": t,
            "count": int(counts_by_type.get(t, 0)),
            "properties": pipe_join(props),  # no duplicates, stable order
        })

    # sort output (e.g., by count desc)
    node_types_properties_counts.sort(key=lambda r: (-r["count"], r["nodeType"]))

    edge_type_counts = aggregate_edge_types(edge_types, rel_type_counts)

    # -------------------------
    # SUMMARY COUNTS (single-row CSV)
    # -------------------------
    summary_counts = {
        "nodes": total_nodes,
        "edges": total_edges,
        "node_types": len(node_types),
        # edge type by your def = relType + OR-aggregated source/target label sets + merged properties
        "edge_types": len(edge_type_counts),
        "node_labels": len(node_labels),
        "edge_labels": len(rel_types),  # relationship types
        "node_patterns": len(node_patterns),
        "edge_patterns": len(edge_patterns),
        "total_properties": len(all_property_keys),
    }

    write_single_row_csv(os.path.join(args.out, "summary_counts.csv"), summary_counts)
    write_json(os.path.join(args.out, "summary_counts.json"), summary_counts)

    # -------------------------
    # CP OPERATIONALIZATION METRICS
    # -------------------------
    node_counts_total = sum(x["count"] for x in node_type_counts)
    unlabeled_nodes = sum(x["count"] for x in node_type_counts if x["nodeType"] == "")

    # CP3: avg label cardinality per node (weighted by node counts per nodeType combo)
    weighted_label_cardinality = 0
    for x in node_type_counts:
        node_type = x["nodeType"]
        cardinality = 0 if node_type == "" else len(node_type.split(":"))
        weighted_label_cardinality += cardinality * x["count"]
    cp3_avg_label_cardinality = (
        (weighted_label_cardinality / node_counts_total) if node_counts_total else 0.0
    )

    # CP1: avg number of distinct node patterns per node type
    node_pattern_counts_by_type = {}
    for row in node_patterns:
        node_pattern_counts_by_type[row["nodeType"]] = node_pattern_counts_by_type.get(row["nodeType"], 0) + 1
    cp1_avg_node_patterns_per_node_type = (
        (sum(node_pattern_counts_by_type.values()) / len(node_pattern_counts_by_type))
        if node_pattern_counts_by_type else 0.0
    )

    # CP5: both typed-global and type-agnostic-global views
    cp5_total_node_patterns_typed = len(node_patterns)  # distinct (nodeType, propSet)
    cp5_global_distinct_property_sets = len(set(row["propSet"] for row in node_patterns))  # distinct propSet only

    # CP7: avg edge patterns per edge type
    cp7_avg_edge_patterns_per_edge_type = (
        (len(edge_patterns) / len(edge_type_counts)) if edge_type_counts else 0.0
    )

    cp_operationalization = {
        "cp1_avg_node_patterns_per_node_type": cp1_avg_node_patterns_per_node_type,
        "cp2_unlabeled_nodes_percent": ((unlabeled_nodes / node_counts_total) * 100.0) if node_counts_total else 0.0,
        "cp3_avg_label_cardinality_per_node": cp3_avg_label_cardinality,
        "cp5_total_node_patterns_typed": cp5_total_node_patterns_typed,
        "cp5_global_distinct_property_sets": cp5_global_distinct_property_sets,
        "cp7_avg_edge_patterns_per_edge_type": cp7_avg_edge_patterns_per_edge_type,
        "cp8_nodes_plus_edges": total_nodes + total_edges,
    }
    write_single_row_csv(os.path.join(args.out, "cp_operationalization.csv"), cp_operationalization)
    write_json(os.path.join(args.out, "cp_operationalization.json"), cp_operationalization)

    # -------------------------
    # EXPORT CSVs
    # -------------------------

    write_csv(os.path.join(args.out, "node_labels.csv"),
              [{"label": l} for l in node_labels],
              ["label"])

    write_csv(os.path.join(args.out, "node_label_counts.csv"),
              node_label_counts,
              ["label", "count"])

    write_csv(os.path.join(args.out, "relationship_types.csv"),
              [{"relType": r} for r in rel_types],
              ["relType"])

    write_csv(os.path.join(args.out, "relationship_type_counts.csv"),
              rel_type_counts,
              ["relType", "count"])

    write_csv(os.path.join(args.out, "node_properties.csv"),
              [{"property": k} for k in node_property_keys],
              ["property"])

    write_csv(os.path.join(args.out, "edge_properties.csv"),
              [{"property": k} for k in edge_property_keys],
              ["property"])

    write_csv(os.path.join(args.out, "all_properties.csv"),
              [{"property": k} for k in all_property_keys],
              ["property"])

    write_csv(os.path.join(args.out, "node_type_counts.csv"),
              node_type_counts,
              ["nodeType", "count"])

    write_csv(os.path.join(args.out, "node_patterns.csv"),
              node_patterns,
              ["nodeType", "propSet", "count"])

    write_csv(os.path.join(args.out, "edge_type_counts.csv"),
              edge_type_counts,
              ["relType", "sourceLabelCombo", "targetLabelCombo", "properties", "count"])

    write_csv(os.path.join(args.out, "edge_patterns.csv"),
              edge_patterns,
              ["relType", "sourceLabelCombo", "targetLabelCombo", "propSet", "count"])

    # NEW FILE:
    write_csv(os.path.join(args.out, "node_types_properties_counts.csv"),
              node_types_properties_counts,
              ["nodeType", "count", "properties"])

    # -------------------------
    # EXPORT JSON (everything)
    # -------------------------
    profile = {
        "summary_counts": summary_counts,
        "cp_operationalization": cp_operationalization,
        "global_stats": {"total_nodes": total_nodes, "total_edges": total_edges},
        "node_labels": node_labels,
        "node_label_counts": node_label_counts,
        "relationship_types": rel_types,
        "relationship_type_counts": rel_type_counts,
        "node_property_keys": node_property_keys,
        "edge_property_keys": edge_property_keys,
        "all_property_keys": all_property_keys,
        "node_types": node_types,
        "node_type_counts": node_type_counts,
        "node_patterns": node_patterns,
        "edge_types": edge_types,
        "edge_type_counts": edge_type_counts,
        "edge_patterns": edge_patterns,
        "node_types_properties_counts": node_types_properties_counts,
    }
    write_json(os.path.join(args.out, "profile.json"), profile)

    print("Export completed. Files written to:", args.out)


if __name__ == "__main__":
    main()
