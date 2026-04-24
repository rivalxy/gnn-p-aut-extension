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
    PseudoSimilarGraph,
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
class GraphContext:
    """Bundles the graph representations needed to build and validate examples."""

    edge_index: torch.Tensor
    num_of_nodes: int
    adjacency_dict: AdjacencyDict
    group: PermutationGroup


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


def _append_validated(
    raw_examples: list[RawPautExample],
    examples: list[tuple[Mapping, int]],
    ctx: GraphContext,
    label: int,
    dataset_type: DatasetType,
    strategy: str,
) -> None:
    """Validate examples against ctx and append them as RawPautExamples."""
    expected_extensible = label == 1
    for mapping, p_aut_size in examples:
        assert is_paut(ctx.adjacency_dict, mapping)
        assert is_extensible(ctx.group, mapping) == expected_extensible

        raw_examples.append(
            RawPautExample(
                edge_index=ctx.edge_index,
                num_of_nodes=ctx.num_of_nodes,
                mapping=mapping,
                label=label,
                paut_stats=PautStats(p_aut_size, label, dataset_type, strategy),
            )
        )


def _build_graph_context(graph_data: GraphData) -> GraphContext:
    """Build a GraphContext from the input graph data by computing its automorphism group."""
    generators_raw, _, _, _, _ = autgrp(graph_data.graph)
    generators = [Permutation(g) for g in generators_raw]
    group = PermutationGroup(generators)
    return GraphContext(
        edge_index=build_edge_index(graph_data.adjacency_dict),
        num_of_nodes=graph_data.num_of_nodes,
        adjacency_dict=graph_data.adjacency_dict,
        group=group,
    )


def _graph_size(group: PermutationGroup, pynauty_graph: pynauty.Graph) -> int:
    """Return the order of the automorphism group of pynauty_graph."""
    _, grpsize1, grpsize2, _, _ = autgrp(pynauty_graph)
    return int(round(grpsize1 * 10**grpsize2))


def _claim_canonical_for_split(
    construction: PseudoSimilarGraph | None,
    seen_canonical: dict[bytes, DatasetType] | None,
    dataset_type: DatasetType,
) -> PseudoSimilarGraph | None:
    """Return construction only if its canonical cert isn't already claimed by a different split.

    Mutates seen_canonical to record the claim.
    """
    if construction is None or seen_canonical is None:
        return construction

    pg_cert = pynauty.Graph(construction.num_nodes)
    pg_cert.set_adjacency_dict(construction.adj)
    cert = pynauty.certificate(pg_cert)
    if cert in seen_canonical and seen_canonical[cert] != dataset_type:
        return None
    seen_canonical[cert] = dataset_type
    return construction


def _constructed_context(construction: PseudoSimilarGraph) -> GraphContext:
    """Build a GraphContext for a Godsil-Kocay constructed graph G."""
    pg = pynauty.Graph(construction.num_nodes)
    pg.set_adjacency_dict(construction.adj)
    gens_raw, _, _, _, _ = pynauty.autgrp(pg)
    gens_G = [Permutation(g) for g in gens_raw] or [
        Permutation(construction.num_nodes - 1)
    ]
    group_G = PermutationGroup(gens_G)
    return GraphContext(
        edge_index=build_edge_index(construction.adj),
        num_of_nodes=construction.num_nodes,
        adjacency_dict=construction.adj,
        group=group_G,
    )


def _emit_positives(
    raw_examples: list[RawPautExample],
    ctx: GraphContext,
    examples_num: int,
    dataset_type: DatasetType,
) -> list[tuple[Mapping, int]]:
    positives = gen_positive_examples(ctx.group, ctx.num_of_nodes, examples_num)
    _append_validated(raw_examples, positives, ctx, 1, dataset_type, "positive")
    return positives


def _emit_constructed_pair(
    raw_examples: list[RawPautExample],
    construction: PseudoSimilarGraph,
    max_negatives: int,
    dataset_type: DatasetType,
) -> tuple[GraphContext, list[tuple[Mapping, int]], list[tuple[Mapping, int]]]:
    """Emit pseudo-similar negatives on G and matching positives on G."""
    ctx_G = _constructed_context(construction)
    negatives = gen_pseudo_similar_examples(
        ctx_G.group,
        ctx_G.num_of_nodes,
        ctx_G.adjacency_dict,
        construction.u,
        construction.v,
        construction.witness,
        max_negatives,
    )
    _append_validated(raw_examples, negatives, ctx_G, 0, dataset_type, "pseudo_similar")

    positives_G: list[tuple[Mapping, int]] = []
    if negatives and not ctx_G.group.is_trivial:
        positives_G = gen_positive_examples(
            ctx_G.group, ctx_G.num_of_nodes, len(negatives)
        )
        _append_validated(raw_examples, positives_G, ctx_G, 1, dataset_type, "positive")

    return ctx_G, negatives, positives_G


def _emit_blocking_fill(
    raw_examples: list[RawPautExample],
    ctx: GraphContext,
    remaining: int,
    dataset_type: DatasetType,
) -> None:
    negatives_blocking = gen_blocking_examples(
        ctx.group, remaining, ctx.num_of_nodes, ctx.adjacency_dict
    )
    _append_validated(
        raw_examples,
        negatives_blocking[:remaining],
        ctx,
        0,
        dataset_type,
        "blocking",
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
        ctx_F = _build_graph_context(graph_data)
        group_order = _graph_size(ctx_F.group, graph_data.graph)
        examples_num = min(max_examples_num, group_order)

        positives = _emit_positives(raw_examples, ctx_F, examples_num, dataset_type)

        construction = find_pseudo_similar_construction(
            ctx_F.adjacency_dict, ctx_F.num_of_nodes, ctx_F.group
        )
        construction = _claim_canonical_for_split(
            construction, seen_canonical, dataset_type
        )

        negatives_constructed: list[tuple[Mapping, int]] = []
        positives_G: list[tuple[Mapping, int]] = []
        if construction is not None:
            _, negatives_constructed, positives_G = _emit_constructed_pair(
                raw_examples, construction, len(positives) // 2, dataset_type
            )

        remaining = len(positives) + len(positives_G) - len(negatives_constructed)
        _emit_blocking_fill(raw_examples, ctx_F, remaining, dataset_type)

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
