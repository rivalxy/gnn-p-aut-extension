import csv
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

import pynauty
import torch
from pynauty import autgrp
from sympy.combinatorics import Permutation, PermutationGroup
from torch_geometric.data import Data

from dataset.features import make_pyg_data
from dataset.graph_utils import (
    AdjacencyDict,
    GraphData,
    Mapping,
    find_pseudo_similar_construction,
    is_extensible,
    is_paut,
)
from dataset.sampling import (
    gen_blocking_examples,
    gen_positive_examples,
    gen_pseudo_similar_examples,
)


class DatasetType(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class PautStats:
    paut_size: int
    label: int
    dataset_type: DatasetType
    strategy: str


def paut_sizes_to_csv(
    stats_by_node_count: dict[int, list[PautStats]], file_path: str
) -> None:
    """Write PautStats grouped by node count to a CSV file.

    :param stats_by_node_count: Dictionary mapping number of nodes to a list of PautStats.
    :param file_path: Path to the output CSV file.
    """
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["num_of_nodes", "paut_size", "label", "dataset_type", "strategy"]
        )
        for num_of_nodes, stats in stats_by_node_count.items():
            for stat in stats:
                writer.writerow(
                    [
                        num_of_nodes,
                        stat.paut_size,
                        stat.label,
                        stat.dataset_type,
                        stat.strategy,
                    ]
                )


@dataclass
class RawPautExample:
    """A raw partial automorphism example before feature encoding."""

    edge_index: torch.Tensor
    num_of_nodes: int
    mapping: Mapping
    label: int
    paut_stats: PautStats


@dataclass(frozen=True)
class DatasetConfiguration:
    """Configuration for encoding and saving one dataset variant."""

    name: str
    raw_train: list[RawPautExample]
    extra_features: bool
    val_paut_sizes: dict[int, list[PautStats]]
    train_output_path: str
    paut_sizes_output_path: str
    val_dataset: list[Data]
    test_dataset: list[Data]


def build_edge_index(adjacency_dict: AdjacencyDict) -> torch.Tensor:
    """Build a PyG edge index tensor from an adjacency dictionary.

    :param adjacency_dict: Dictionary mapping each node to its set of neighbors.
    :returns: Long tensor of shape (2, num_edges) suitable for PyG.
    """
    if len(adjacency_dict) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edges = []
    for u, neighbors in adjacency_dict.items():
        for v in neighbors:
            edges.append([u, v])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def append_validated_examples(
    raw_examples: list[RawPautExample],
    examples: list[tuple[Mapping, int]],
    *,
    edge_index: torch.Tensor,
    num_of_nodes: int,
    adjacency_dict: AdjacencyDict,
    group: PermutationGroup,
    label: int,
    dataset_type: DatasetType,
    strategy: str,
) -> None:
    """Validate examples and append them to raw_examples.

    Asserts that each mapping is a valid partial automorphism and that its
    extensibility matches the given label, then appends a RawPautExample.

    :param raw_examples: List to append validated examples to.
    :param examples: List of (mapping, paut_size) tuples to validate and append.
    :param edge_index: PyG edge index tensor for the graph.
    :param num_of_nodes: Number of nodes in the graph.
    :param adjacency_dict: Adjacency dictionary of the graph.
    :param group: Automorphism group of the graph.
    :param label: Expected label (1 = extensible, 0 = non-extensible).
    :param dataset_type: Which split (train/val/test) this example belongs to.
    :param strategy: Sampling strategy used to generate these examples.
    """
    expected_extensible = label == 1
    for mapping, p_aut_size in examples:
        assert is_paut(adjacency_dict, mapping)
        assert is_extensible(group, mapping) == expected_extensible

        raw_examples.append(
            RawPautExample(
                edge_index=edge_index,
                num_of_nodes=num_of_nodes,
                mapping=mapping,
                label=label,
                paut_stats=PautStats(p_aut_size, label, dataset_type, strategy),
            )
        )


def generate_raw_examples(
    pynauty_graphs: list[GraphData],
    dataset_type: DatasetType,
    max_examples_num: int,
    seen_canonical: dict[bytes, DatasetType] | None = None,
) -> list[RawPautExample]:
    """Generate raw partial automorphism examples without feature encoding.

    :param pynauty_graphs: Source graphs to generate examples from.
    :param dataset_type: Which split (train/val/test) these examples belong to.
    :param max_examples_num: Maximum examples per graph.
    :param seen_canonical: Optional shared dict mapping nauty canonical certificates
        to the split that first used that constructed graph G. When provided, any G
        whose certificate was already claimed by a different split is skipped, preventing
        isomorphic constructed graphs from appearing in multiple splits.
    """
    raw_examples: list[RawPautExample] = []

    for graph_data in pynauty_graphs:
        pynauty_graph = graph_data.graph
        num_of_nodes = graph_data.num_of_nodes
        adjacency_dict = graph_data.adjacency_dict
        generators_raw, grpsize1, grpsize2, _, _ = autgrp(pynauty_graph)
        group_size = int(round(grpsize1 * 10**grpsize2))

        examples_num = min(max_examples_num, group_size)
        generators = [Permutation(g) for g in generators_raw]
        group = PermutationGroup(generators)

        tensor_edge_index = build_edge_index(adjacency_dict)

        # --- positives from F ---
        positives = gen_positive_examples(group, num_of_nodes, examples_num)
        append_validated_examples(
            raw_examples,
            positives,
            edge_index=tensor_edge_index,
            num_of_nodes=num_of_nodes,
            adjacency_dict=adjacency_dict,
            group=group,
            label=1,
            dataset_type=dataset_type,
            strategy="positive",
        )

        # --- negatives: up to 50% from Godsil-Kocay construction, rest blocking ---
        max_constructed = len(positives) // 2
        negatives_constructed: list[tuple[Mapping, int]] = []
        constructed_edge_index = tensor_edge_index
        constructed_adj = adjacency_dict
        constructed_num_nodes = num_of_nodes
        constructed_group = group

        construction = find_pseudo_similar_construction(
            adjacency_dict, num_of_nodes, group
        )

        # Guard against isomorphic G's appearing in multiple splits.
        if construction is not None and seen_canonical is not None:
            pg_cert = pynauty.Graph(construction.num_nodes)
            pg_cert.set_adjacency_dict(construction.adj)
            cert = pynauty.certificate(pg_cert)
            if cert in seen_canonical and seen_canonical[cert] != dataset_type:
                construction = None
            else:
                seen_canonical[cert] = dataset_type
        if construction is not None:
            adj_G, n_G, u, v, witness = construction
            pg = pynauty.Graph(n_G)
            pg.set_adjacency_dict(adj_G)
            gens_raw, _, _, _, _ = pynauty.autgrp(pg)
            gens_G = [Permutation(g) for g in gens_raw] or [Permutation(n_G - 1)]
            group_G = PermutationGroup(gens_G)

            negatives_constructed = gen_pseudo_similar_examples(
                group_G, n_G, adj_G, u, v, witness, max_constructed
            )
            constructed_edge_index = build_edge_index(adj_G)
            constructed_adj = adj_G
            constructed_num_nodes = n_G
            constructed_group = group_G

        # Append constructed pseudo-similar negatives (on graph G)
        append_validated_examples(
            raw_examples,
            negatives_constructed,
            edge_index=constructed_edge_index,
            num_of_nodes=constructed_num_nodes,
            adjacency_dict=constructed_adj,
            group=constructed_group,
            label=0,
            dataset_type=dataset_type,
            strategy="pseudo_similar",
        )

        # Generate matching positives from G so the model sees both labels
        # for the constructed graph structure (only if G has non-trivial automorphisms)
        positives_G: list[tuple[Mapping, int]] = []
        if negatives_constructed and not constructed_group.is_trivial:
            positives_G = gen_positive_examples(
                constructed_group, constructed_num_nodes, len(negatives_constructed)
            )
            append_validated_examples(
                raw_examples,
                positives_G,
                edge_index=constructed_edge_index,
                num_of_nodes=constructed_num_nodes,
                adjacency_dict=constructed_adj,
                group=constructed_group,
                label=1,
                dataset_type=dataset_type,
                strategy="positive",
            )

        # Fill blocking negatives so total negatives == total positives
        remaining = len(positives) + len(positives_G) - len(negatives_constructed)
        negatives_blocking = gen_blocking_examples(
            group, remaining, num_of_nodes, adjacency_dict
        )
        append_validated_examples(
            raw_examples,
            negatives_blocking[:remaining],
            edge_index=tensor_edge_index,
            num_of_nodes=num_of_nodes,
            adjacency_dict=adjacency_dict,
            group=group,
            label=0,
            dataset_type=dataset_type,
            strategy="blocking",
        )

    return raw_examples


def raw_examples_to_pyg(
    raw_examples: list[RawPautExample], extra_features: bool
) -> tuple[list[Data], dict[int, list[PautStats]]]:
    """Convert raw examples to PyG Data objects for one feature variant."""
    pyg_data = []
    paut_sizes: dict[int, list[PautStats]] = defaultdict(list)

    for example in raw_examples:
        paut_sizes[example.num_of_nodes].append(example.paut_stats)
        pyg_data.append(
            make_pyg_data(
                example.edge_index,
                example.num_of_nodes,
                example.mapping,
                example.label,
                extra_features,
            )
        )

    return pyg_data, paut_sizes
