#!/usr/bin/env python3

import argparse
import csv
import os
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a loose .pgs schema from node and edge type stats."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name. The script reads from stats_<dataset>/.",
    )
    parser.add_argument(
        "--stats-dir",
        help="Optional explicit stats directory. Defaults to stats_<dataset>/.",
    )
    parser.add_argument(
        "--graph-name",
        help="Optional graph type name. Defaults to <dataset>_GraphType.",
    )
    parser.add_argument(
        "--out",
        help="Optional output file. Defaults to stats_<dataset>/<dataset>.pgs.",
    )
    return parser.parse_args()


def quote_name(value: str) -> str:
    return value


def sanitize_type_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return sanitized or "type"


def label_spec_from_combo(label_combo: str) -> str:
    labels = [part.strip() for part in label_combo.split(":") if part.strip()]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return " & ".join(labels)


def label_spec_from_or_combos(value: str) -> str:
    combos = [combo.strip() for combo in value.split("|") if combo.strip()]
    specs = []
    for combo in combos:
        spec = label_spec_from_combo(combo)
        if not spec:
            continue
        if " & " in spec:
            specs.append(f"( {spec} )")
        else:
            specs.append(spec)
    return " | ".join(specs)


def read_csv_rows(path: str):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def property_spec(properties):
    cleaned = [prop.strip() for prop in properties if prop and prop.strip()]
    if not cleaned:
        return ""
    props = ", ".join(f"{quote_name(prop)} STRING" for prop in sorted(set(cleaned)))
    return f"{{ {props} }}"


def build_node_types(rows):
    node_type_defs = []
    node_type_names = []

    for row in rows:
        node_type = row["nodeType"].strip()
        if not node_type:
            continue

        type_name = f"node_{sanitize_type_name(node_type)}"
        label_spec = label_spec_from_combo(node_type)
        properties = row.get("properties", "").split("|")
        node_props = property_spec(properties)
        definition = (
            f'CREATE NODE TYPE {quote_name(type_name)} '
            f'( {quote_name(type_name)} : {label_spec}{f" {node_props}" if node_props else ""} );'
        )
        node_type_defs.append(definition)
        node_type_names.append(type_name)

    return node_type_defs, node_type_names


def build_edge_types(rows):
    edge_type_defs = []
    edge_type_names = []

    for row in rows:
        rel_type = row["relType"].strip()
        source_combo = row["sourceLabelCombo"].strip()
        target_combo = row["targetLabelCombo"].strip()
        if not rel_type or not source_combo or not target_combo:
            continue

        type_name = f"edge_{sanitize_type_name(rel_type)}"
        source_label_spec = label_spec_from_or_combos(source_combo)
        target_label_spec = label_spec_from_or_combos(target_combo)
        edge_props = property_spec(row.get("properties", "").split("|"))
        definition = (
            f'CREATE EDGE TYPE {quote_name(type_name)} '
            f'( : {source_label_spec} ) - '
            f'[ {quote_name(type_name)} : {quote_name(rel_type)}{f" {edge_props}" if edge_props else ""} ] -> '
            f'( : {target_label_spec} );'
        )
        edge_type_defs.append(definition)
        edge_type_names.append(type_name)

    return edge_type_defs, edge_type_names


def build_graph_type(graph_name: str, element_type_names):
    if not element_type_names:
        return f"CREATE GRAPH TYPE {quote_name(graph_name)} STRICT {{}};"
    elements = ",\n  ".join(quote_name(name) for name in element_type_names)
    return f"CREATE GRAPH TYPE {quote_name(graph_name)} STRICT {{\n  {elements}\n}};"


def main():
    args = parse_args()

    stats_dir = args.stats_dir or f"stats_{args.dataset}"
    node_types_path = os.path.join(stats_dir, "node_types_properties_counts.csv")
    edge_types_path = os.path.join(stats_dir, "edge_type_counts.csv")
    graph_name = args.graph_name or f"{args.dataset}_GraphType"
    out_path = args.out or os.path.join(stats_dir, f"{args.dataset}.pgs")

    if not os.path.exists(node_types_path):
        raise FileNotFoundError(f"Missing node type stats: {node_types_path}")
    if not os.path.exists(edge_types_path):
        raise FileNotFoundError(f"Missing edge type stats: {edge_types_path}")

    node_rows = read_csv_rows(node_types_path)
    edge_rows = read_csv_rows(edge_types_path)

    node_type_defs, node_type_names = build_node_types(node_rows)
    edge_type_defs, edge_type_names = build_edge_types(edge_rows)
    graph_type_def = build_graph_type(graph_name, node_type_names + edge_type_names)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    schema = "\n".join(node_type_defs + edge_type_defs + [graph_type_def]) + "\n"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(schema)

    print(out_path)


if __name__ == "__main__":
    main()
