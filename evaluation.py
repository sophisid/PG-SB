import os
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

import pandas as pd


@dataclass
class Metrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def _compute_metrics_from_counts(tp: int, fp: int, fn: int) -> Metrics:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return Metrics(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def compute_node_metrics_from_csv(
    original_nodes_csv: str,
    predicted_nodes_csv: str,
) -> Metrics:
    """
    Expects:
      - original_nodes_csv: columns ['_nodeId', 'original_label']
      - predicted_nodes_csv: columns ['merged_cluster_id', 'sortedLabels', 'nodeIdsInCluster']
        where 'sortedLabels' is colon-separated string and 'nodeIdsInCluster' is ';'-separated ids.
    """

    if not os.path.exists(original_nodes_csv) or not os.path.exists(predicted_nodes_csv):
        print(f"[ERROR] Node CSVs not found: {original_nodes_csv}, {predicted_nodes_csv}")
        return Metrics(0, 0, 0, 0.0, 0.0, 0.0)

    orig = pd.read_csv(original_nodes_csv)
    pred = pd.read_csv(predicted_nodes_csv)

    if "original_label" not in orig.columns:
        print("[ERROR] Column 'original_label' is missing from original_nodes_csv.")
        return Metrics(0, 0, 0, 0.0, 0.0, 0.0)

    # Assume nodeIdsInCluster like "1;2;3"
    pred = pred.copy()
    pred["nodeIdsInCluster"] = pred["nodeIdsInCluster"].fillna("").astype(str)

    pred_rows = []
    for _, row in pred.iterrows():
        cluster_id = row["merged_cluster_id"]
        sorted_labels = str(row["sortedLabels"]) if pd.notna(row["sortedLabels"]) else ""
        predicted_labels = [l for l in sorted_labels.split(":") if l]

        node_ids_raw = str(row["nodeIdsInCluster"])
        node_ids = [nid for nid in node_ids_raw.split(";") if nid]

        for nid in node_ids:
            try:
                nid_long = int(nid)
            except ValueError:
                continue
            pred_rows.append(
                {
                    "nodeId": nid_long,
                    "predictedLabels": list(dict.fromkeys(predicted_labels)),  # unique, preserve order
                    "merged_cluster_id": cluster_id,
                }
            )

    if not pred_rows:
        print("[WARN] No predicted nodes after exploding; returning zeros.")
        return Metrics(0, 0, 0, 0.0, 0.0, 0.0)

    exploded_pred = pd.DataFrame(pred_rows)

    orig = orig.copy()
    orig["_nodeId"] = orig["_nodeId"].astype("int64")
    orig["original_label"] = orig["original_label"].fillna("").astype(str)

    def split_labels(x: str) -> List[str]:
        return [l for l in x.split(",") if l]

    orig["actualLabels"] = orig["original_label"].apply(split_labels)
    exploded_orig = orig[["_nodeId", "actualLabels"]].rename(columns={"_nodeId": "nodeId"})

    evaluation = exploded_pred.merge(exploded_orig, on="nodeId", how="left")

    # Majority logic
    # Find majority label per cluster based on actualLabels
    merged = exploded_pred.merge(exploded_orig, on="nodeId", how="inner")
    merged["actualLabels"] = merged["actualLabels"].apply(lambda x: x if isinstance(x, list) else [])
    merged = merged.explode("actualLabels")
    merged = merged[merged["actualLabels"].notna() & (merged["actualLabels"] != "")]

    if merged.empty:
        print("[WARN] No overlapping nodes with actualLabels; returning zeros.")
        return Metrics(0, 0, 0, 0.0, 0.0, 0.0)

    freq = (
        merged.groupby(["merged_cluster_id", "actualLabels"])
        .size()
        .reset_index(name="freq")
    )

    # For each cluster, keep the label with max freq
    freq["rank"] = freq.groupby("merged_cluster_id")["freq"].rank(
        method="first", ascending=False
    )
    majority_df = freq[freq["rank"] == 1][["merged_cluster_id", "actualLabels"]]
    majority_df = majority_df.rename(columns={"actualLabels": "majority_label"})

    evaluation["actualLabels"] = evaluation["actualLabels"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    eval_with_majority = evaluation.merge(majority_df, on="merged_cluster_id", how="inner")

    def has_majority(row):
        return 1 if row["majority_label"] in row["actualLabels"] else 0

    eval_with_majority["correctAssignmentMajority"] = eval_with_majority.apply(
        has_majority, axis=1
    )

    tp = int((eval_with_majority["correctAssignmentMajority"] == 1).sum())
    fp = int((eval_with_majority["correctAssignmentMajority"] == 0).sum())

    exploded_orig2 = exploded_orig.copy()
    exploded_orig2["actualLabelsKey"] = exploded_orig2["actualLabels"].apply(tuple)
    total_actual = (
        exploded_orig2.groupby("actualLabelsKey").size().reset_index(name="totalActual")
    )

    pos_pred = eval_with_majority[eval_with_majority["correctAssignmentMajority"] == 1].copy()
    pos_pred["actualLabelsKey"] = pos_pred["actualLabels"].apply(tuple)
    total_pred = (
        pos_pred.groupby("actualLabelsKey").size().reset_index(name="totalPredicted")
    )

    actual_pred = total_actual.merge(total_pred, on="actualLabelsKey", how="left").fillna(0)
    actual_pred["totalActual"] = actual_pred["totalActual"].astype(int)
    actual_pred["totalPredicted"] = actual_pred["totalPredicted"].astype(int)

    actual_pred["fnPerGroup"] = actual_pred.apply(
        lambda r: max(r["totalActual"] - r["totalPredicted"], 0), axis=1
    )
    fn = int(actual_pred["fnPerGroup"].sum())

    metrics = _compute_metrics_from_counts(tp, fp, fn)

    print("\nMajority Label Node Evaluation Metrics:")
    print(f"  TP: {tp}")
    print(f"  FP: {fp}")
    print(f"  FN: {fn}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1-Score:  {metrics.f1:.4f}")

    return metrics


def compute_edge_metrics_from_csv(
    original_edges_csv: str,
    predicted_edges_csv: str,
) -> Metrics:
    """
    Expects:
      - original_edges_csv: ['srcId','dstId','relationshipType','srcType','dstType']
      - predicted_edges_csv: ['merged_cluster_id','relationshipTypes','srcLabels','dstLabels','edgeIdsInCluster']
        where relationshipTypes, srcLabels, dstLabels are comma-separated,
        and edgeIdsInCluster is 'srcId|dstId;srcId|dstId;...'.
    """
    if not os.path.exists(original_edges_csv) or not os.path.exists(predicted_edges_csv):
        print(f"[ERROR] Edge CSVs not found: {original_edges_csv}, {predicted_edges_csv}")
        return Metrics(0, 0, 0, 0.0, 0.0, 0.0)

    orig = pd.read_csv(original_edges_csv)
    pred = pd.read_csv(predicted_edges_csv)

    required = {"relationshipType", "srcType", "dstType", "srcId", "dstId"}
    missing = required - set(orig.columns)
    if missing:
        print(f"[ERROR] Missing required columns in original_edges_csv: {missing}")
        return Metrics(0, 0, 0, 0.0, 0.0, 0.0)

    orig = orig.copy()
    orig["srcId"] = orig["srcId"].astype("int64")
    orig["dstId"] = orig["dstId"].astype("int64")

    orig["edgeId"] = list(zip(orig["srcId"], orig["dstId"]))

    def to_arr(x):
        return [x] if pd.notna(x) and x != "" else []

    orig["actualRelationshipTypes"] = orig["relationshipType"].apply(to_arr)
    orig["actualSrcLabels"] = orig["srcType"].apply(to_arr)
    orig["actualDstLabels"] = orig["dstType"].apply(to_arr)

    exploded_orig = orig[["edgeId", "actualRelationshipTypes", "actualSrcLabels", "actualDstLabels"]]

    pred = pred.copy()

    def split_csv(s: str) -> List[str]:
        return [x for x in s.split(",") if x]

    def parse_edge_ids(s: str) -> List[Tuple[int, int]]:
        if pd.isna(s):
            return []
        parts = str(s).split(";")
        res = []
        for p in parts:
            if "|" not in p:
                continue
            src, dst = p.split("|", 1)
            try:
                res.append((int(src), int(dst)))
            except ValueError:
                continue
        return res

    pred["relationshipTypes"] = pred["relationshipTypes"].fillna("").astype(str).apply(split_csv)
    pred["srcLabels"] = pred["srcLabels"].fillna("").astype(str).apply(split_csv)
    pred["dstLabels"] = pred["dstLabels"].fillna("").astype(str).apply(split_csv)
    pred["edgeIdsInCluster"] = pred["edgeIdsInCluster"].fillna("").astype(str).apply(parse_edge_ids)

    pred_rows = []
    for _, row in pred.iterrows():
        cluster_id = row["merged_cluster_id"]
        rel_types = list(dict.fromkeys(row["relationshipTypes"]))
        src_labels = list(dict.fromkeys(row["srcLabels"]))
        dst_labels = list(dict.fromkeys(row["dstLabels"]))
        for (src, dst) in row["edgeIdsInCluster"]:
            pred_rows.append(
                {
                    "edgeId": (src, dst),
                    "predictedRelationshipTypes": rel_types,
                    "predictedSrcLabels": src_labels,
                    "predictedDstLabels": dst_labels,
                    "merged_cluster_id": cluster_id,
                }
            )

    if not pred_rows:
        print("[WARN] No predicted edges after exploding; returning zeros.")
        return Metrics(0, 0, 0, 0.0, 0.0, 0.0)

    exploded_pred = pd.DataFrame(pred_rows)

    evaluation = exploded_pred.merge(exploded_orig, on="edgeId", how="inner")

    merged = exploded_pred.merge(exploded_orig, on="edgeId", how="inner")
    merged["actualRelationshipTypes"] = merged["actualRelationshipTypes"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    merged = merged.explode("actualRelationshipTypes")
    merged = merged[merged["actualRelationshipTypes"].notna() & (merged["actualRelationshipTypes"] != "")]

    if merged.empty:
        print("[WARN] No overlapping edges with actualRelationshipTypes; returning zeros.")
        return Metrics(0, 0, 0, 0.0, 0.0, 0.0)

    freq = (
        merged.groupby(["merged_cluster_id", "actualRelationshipTypes"])
        .size()
        .reset_index(name="freq")
    )

    freq["rank"] = freq.groupby("merged_cluster_id")["freq"].rank(
        method="first", ascending=False
    )
    majority_df = freq[freq["rank"] == 1][["merged_cluster_id", "actualRelationshipTypes"]]
    majority_df = majority_df.rename(
        columns={"actualRelationshipTypes": "majority_relationship_type"}
    )

    evaluation["actualRelationshipTypes"] = evaluation[
        "actualRelationshipTypes"
    ].apply(lambda x: x if isinstance(x, list) else [])

    eval_with_majority = evaluation.merge(majority_df, on="merged_cluster_id", how="inner")

    def has_majority_rel(row):
        return 1 if row["majority_relationship_type"] in row["actualRelationshipTypes"] else 0

    eval_with_majority["correctAssignmentMajority"] = eval_with_majority.apply(
        has_majority_rel, axis=1
    )

    tp = int((eval_with_majority["correctAssignmentMajority"] == 1).sum())
    fp = int((eval_with_majority["correctAssignmentMajority"] == 0).sum())

    exploded_orig2 = exploded_orig.copy()
    exploded_orig2["actualRelationshipTypesKey"] = exploded_orig2["actualRelationshipTypes"].apply(tuple)
    exploded_orig2["actualSrcLabelsKey"] = exploded_orig2["actualSrcLabels"].apply(tuple)
    exploded_orig2["actualDstLabelsKey"] = exploded_orig2["actualDstLabels"].apply(tuple)

    total_actual = (
        exploded_orig2.groupby(
            ["actualRelationshipTypesKey", "actualSrcLabelsKey", "actualDstLabelsKey"]
        )
        .size()
        .reset_index(name="totalActual")
    )

    pos_pred = eval_with_majority[eval_with_majority["correctAssignmentMajority"] == 1].copy()
    pos_pred["actualRelationshipTypesKey"] = pos_pred["actualRelationshipTypes"].apply(tuple)
    pos_pred["actualSrcLabelsKey"] = eval_with_majority["actualSrcLabels"].apply(
        lambda x: tuple(x) if isinstance(x, list) else tuple()
    )
    pos_pred["actualDstLabelsKey"] = eval_with_majority["actualDstLabels"].apply(
        lambda x: tuple(x) if isinstance(x, list) else tuple()
    )

    total_pred = (
        pos_pred.groupby(
            ["actualRelationshipTypesKey", "actualSrcLabelsKey", "actualDstLabelsKey"]
        )
        .size()
        .reset_index(name="totalPredicted")
    )

    actual_pred = total_actual.merge(
        total_pred,
        on=["actualRelationshipTypesKey", "actualSrcLabelsKey", "actualDstLabelsKey"],
        how="left",
    ).fillna(0)
    actual_pred["totalActual"] = actual_pred["totalActual"].astype(int)
    actual_pred["totalPredicted"] = actual_pred["totalPredicted"].astype(int)

    actual_pred["fnPerGroup"] = actual_pred.apply(
        lambda r: max(r["totalActual"] - r["totalPredicted"], 0), axis=1
    )
    fn = int(actual_pred["fnPerGroup"].sum())

    metrics = _compute_metrics_from_counts(tp, fp, fn)

    print("\nMajority Label Edge Evaluation Metrics:")
    print(f"  TP: {tp}")
    print(f"  FP: {fp}")
    print(f"  FN: {fn}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1-Score:  {metrics.f1:.4f}")

    return metrics



def write_run_metrics_json(
    output_dir: str,
    dataset: str,
    method: str,
    noise: int,
    label_percent: float,
    node_metrics: Metrics,
    edge_metrics: Metrics,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    filename = (
        f"metrics_{dataset.upper()}_noise{noise}_labels{label_percent:.2f}_{method.upper()}.json"
    ).replace(",", ".")
    path = os.path.join(output_dir, filename)

    payload = {
        "dataset": dataset,
        "method": method,
        "noise": noise,
        "label_percent": label_percent,
        "nodes": asdict(node_metrics),
        "edges": asdict(edge_metrics),
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] Metrics JSON written to: {path}")
    return path


def evaluate_run_from_csvs(
    output_dir: str,
    dataset: str,
    method: str,
    noise: int,
    label_percent: float,
    original_nodes_csv: str,
    predicted_nodes_csv: str,
    original_edges_csv: str,
    predicted_edges_csv: str,
) -> str:
    node_metrics = compute_node_metrics_from_csv(original_nodes_csv, predicted_nodes_csv)
    edge_metrics = compute_edge_metrics_from_csv(original_edges_csv, predicted_edges_csv)

    json_path = write_run_metrics_json(
        output_dir=output_dir,
        dataset=dataset,
        method=method,
        noise=noise,
        label_percent=label_percent,
        node_metrics=node_metrics,
        edge_metrics=edge_metrics,
    )
    return json_path
