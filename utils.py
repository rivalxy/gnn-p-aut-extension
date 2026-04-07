import json
from pathlib import Path
from typing import Any

import pandas as pd
import pynauty
import torch
import torch.nn as nn
import torch_geometric.data
import torch_geometric.loader
import torch_geometric.utils
from sklearn import metrics
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from dataset.graph_utils import build_adjacency_dict
from models import GIN, GPS

FEATURE_TARGET_ID = 1
FEATURE_SOURCE_ID = 2
PE_DIM = 5


def paut_size_from_torch(torch_graph: torch_geometric.data.Data) -> int:
    x = torch_graph.x
    if x is None:
        return 0
    paut_size = int((x[:, FEATURE_TARGET_ID] != -1).sum().item())
    assert (x[:, FEATURE_TARGET_ID] != -1).sum() == (
        x[:, FEATURE_SOURCE_ID] != -1
    ).sum()
    return paut_size


def aut_grp_size_from_torch(torch_graph: torch_geometric.data.Data) -> int:
    num_nodes = torch_graph.num_nodes
    nx_graph = torch_geometric.utils.to_networkx(torch_graph)
    pynauty_graph = pynauty.Graph(num_nodes, directed=False)
    adjacency_dict = build_adjacency_dict(nx_graph.edges())
    pynauty_graph.set_adjacency_dict(adjacency_dict)
    _, grpsize1, grpsize2, _, _ = pynauty.autgrp(pynauty_graph)
    return grpsize1 * 10**grpsize2


def regularity_check(graph: torch_geometric.data.Data) -> bool:
    if graph.edge_index is None:
        return True
    degrees = torch_geometric.utils.degree(
        graph.edge_index[0], num_nodes=graph.num_nodes
    )
    return bool(torch.all(degrees == degrees[0]).item())


def load_or_compute_dataset_metadata(
    dataset: list[torch_geometric.data.Data], dataset_path: str
) -> list[dict[str, Any]]:
    """Compute per-graph metadata (aut_grp_size, regularity, etc.) with file caching.

    The cache is stored next to the dataset file as ``<name>_metadata_cache.json``.
    """
    cache_path = Path(dataset_path).with_name(
        Path(dataset_path).stem + "_metadata_cache.json"
    )

    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) == len(dataset):
            return cached

    metadata = []
    for graph in dataset:
        metadata.append(
            {
                "num_nodes": graph.num_nodes,
                "regular": regularity_check(graph),
                "paut_size": paut_size_from_torch(graph),
                "aut_grp_size": aut_grp_size_from_torch(graph),
            }
        )

    with open(cache_path, "w") as f:
        json.dump(metadata, f)

    return metadata


def evaluate_checkpoint(
    config_path: str, dataset_path: str, checkpoint_path: str
) -> dict[str, Any]:
    config = load_json(config_path)

    evaluation_dataset = torch.load(dataset_path, weights_only=False)

    dataset_metadata = load_or_compute_dataset_metadata(
        evaluation_dataset, dataset_path
    )

    is_gps = "num_heads" in config
    if is_gps:
        pe_transform = AddLaplacianEigenvectorPE(
            k=PE_DIM,
            attr_name="laplacian_eigenvector_pe",
            is_undirected=True,
        )
        evaluation_dataset = [pe_transform(data) for data in evaluation_dataset]

    evaluation_loader = torch_geometric.loader.DataLoader(
        evaluation_dataset, batch_size=config["batch_size"], shuffle=False
    )

    number_of_features = evaluation_dataset[0].num_node_features
    if is_gps:
        evaluation_model: nn.Module = GPS(
            number_of_features,
            config["hidden_dim"],
            config["num_layers"],
            config["dropout"],
            config["num_heads"],
            PE_DIM,
        )
    else:
        evaluation_model = GIN(
            number_of_features,
            config["hidden_dim"],
            config["num_layers"],
            config["dropout"],
        )
    evaluation_model.load_state_dict(
        torch.load(checkpoint_path, map_location=torch.device("cpu"))
    )
    evaluation_model.eval()

    records, true_labels, predictions = collect_prediction_records(
        evaluation_model, evaluation_loader, dataset_metadata
    )
    predictions_df = build_predictions_df(records)

    return {
        "model": evaluation_model,
        "dataset": evaluation_dataset,
        "loader": evaluation_loader,
        "predictions_df": predictions_df,
        "accuracy": predictions_df["correct"].mean(),
        "f1": metrics.f1_score(true_labels, predictions, zero_division=0),
        "config": config,
    }


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def collect_prediction_records(
    model: nn.Module,
    loader: torch_geometric.loader.DataLoader,
    dataset_metadata: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    records: list[dict[str, Any]] = []
    true_labels: list[int] = []
    predictions: list[int] = []
    sample_idx = 0

    for batch in loader:
        with torch.no_grad():
            logits = model(batch).view(-1)
            probs = torch.sigmoid(logits)
            pred = (logits > 0).float()

        for i, graph in enumerate(batch.to_data_list()):
            true_label = int(graph.y.item())
            pred_label = int(pred[i].item())
            meta = dataset_metadata[sample_idx]
            records.append(
                {
                    "sample_idx": sample_idx,
                    "num_nodes": meta["num_nodes"],
                    "regular": meta["regular"],
                    "paut_size": meta["paut_size"],
                    "aut_grp_size": meta["aut_grp_size"],
                    "true_label": true_label,
                    "prediction": pred_label,
                    "pred_prob": float(probs[i].item()),
                    "correct": pred_label == true_label,
                }
            )
            true_labels.append(true_label)
            predictions.append(pred_label)
            sample_idx += 1

    return records, true_labels, predictions


def build_predictions_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    predictions_df = pd.DataFrame(records)
    predictions_df["paut_relative_size"] = (
        predictions_df["paut_size"] / predictions_df["num_nodes"]
    )
    predictions_df["error"] = (
        predictions_df["true_label"] != predictions_df["prediction"]
    ).astype(int)
    return predictions_df
