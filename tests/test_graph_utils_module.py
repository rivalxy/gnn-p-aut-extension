from pathlib import Path

import networkx as nx
import pytest
from sympy.combinatorics import Permutation, PermutationGroup

from dataset.build import DatasetType, PautStats, paut_sizes_to_csv
from dataset.graph_utils import (
    bfs_expand_pseudo_similar,
    build_adjacency_dict,
    build_orbit_map,
    construct_pseudo_similar_graph,
    find_pseudo_similar_construction,
    is_extensible,
    is_injective,
    is_paut,
    read_graphs_from_g6,
)


def test_read_graphs_from_g6(tmp_path: Path) -> None:
    g1 = nx.Graph()
    g1.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4)])

    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (1, 2)])

    g6_path = tmp_path / "test_graphs.g6"
    with open(g6_path, "wb") as f:
        f.write(nx.to_graph6_bytes(g1, header=False))
        f.write(nx.to_graph6_bytes(g2, header=False))

    graphs = read_graphs_from_g6(str(g6_path))
    assert len(graphs) == 2
    assert graphs[0].num_of_nodes == 5
    assert graphs[0].adjacency_dict == {0: {1, 2}, 1: {0, 3}, 2: {0, 4}, 3: {1}, 4: {2}}
    assert graphs[1].num_of_nodes == 3


@pytest.mark.parametrize(
    "mapping, expected",
    [
        ({0: 0, 1: 1, 2: 2}, True),
        ({0: 0, 1: 1, 4: 4}, True),
        ({0: 2, 1: 1, 2: 0}, True),
        ({0: 0, 1: 1, 2: 2, 3: 4, 4: 3}, True),
        ({0: 3, 1: 1, 2: 4}, False),
        ({0: 1, 5: 6}, True),
        ({0: 1, 1: 0, 5: 6, 6: 5}, True),
        ({0: 5, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0}, False),
        (dict(), False),
    ],
)
def test_is_paut(mapping: dict[int, int], expected: bool) -> None:
    adjacency_dict = {
        0: {1},
        1: {0, 2},
        2: {1, 3, 4},
        3: {2, 4},
        4: {2, 3, 5},
        5: {4, 6},
        6: {5},
    }
    assert is_paut(adjacency_dict, mapping) == expected


@pytest.fixture(scope="module")
def path_graph_group() -> PermutationGroup:
    from pynauty import Graph, autgrp

    adjacency_dict = {
        0: {1},
        1: {0, 2},
        2: {1, 3, 4},
        3: {2, 4},
        4: {2, 3, 5},
        5: {4, 6},
        6: {5},
    }
    graph = Graph(7)
    graph.set_adjacency_dict(adjacency_dict)
    generators_raw = autgrp(graph)[0]
    generators = [Permutation(g) for g in generators_raw]
    return PermutationGroup(generators)


@pytest.mark.parametrize(
    "mapping, expected",
    [
        ({0: 0, 1: 1, 2: 2}, True),
        ({0: 0, 1: 1, 4: 4}, True),
        ({0: 2, 1: 1, 2: 0}, False),
        ({0: 0, 1: 1, 2: 2, 3: 4, 4: 3}, False),
        ({}, True),
        ({0: 1, 5: 6}, False),
        ({0: 1, 1: 0, 5: 6, 6: 5}, False),
    ],
)
def test_is_extensible(
    path_graph_group: PermutationGroup, mapping: dict[int, int], expected: bool
) -> None:
    assert is_extensible(path_graph_group, mapping) == expected


# --- is_injective ---


@pytest.mark.parametrize(
    "mapping, expected",
    [
        ({0: 1, 1: 2, 2: 3}, True),
        ({0: 1, 1: 1}, False),
        ({5: 5}, True),
        ({}, True),
    ],
)
def test_is_injective(mapping: dict[int, int], expected: bool) -> None:
    assert is_injective(mapping) == expected


# --- build_adjacency_dict ---


def test_build_adjacency_dict_from_edges() -> None:
    adj = build_adjacency_dict([(0, 1), (1, 2)])
    assert adj == {0: {1}, 1: {0, 2}, 2: {1}}


def test_build_adjacency_dict_empty() -> None:
    assert build_adjacency_dict([]) == {}


# --- build_orbit_map ---


def test_build_orbit_map(path_graph_group: PermutationGroup) -> None:
    orbit_of = build_orbit_map(path_graph_group)
    # Automorphism is (0 6)(1 5)(2 4), so:
    assert orbit_of[0] == orbit_of[6]  # swapped
    assert orbit_of[1] == orbit_of[5]  # swapped
    assert orbit_of[2] == orbit_of[4]  # swapped
    assert orbit_of[3] != orbit_of[0]  # 3 is a fixed point, different orbit


# --- bfs_expand_pseudo_similar ---
# TODO


def test_bfs_expand_pseudo_similar_contains_seed() -> None:
    adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    # sigma is an isomorphism V(G-0) -> V(G-3): the reversed path {1,2,3} -> {0,1,2}
    sigma = {1: 2, 2: 1, 3: 0}
    mapping = bfs_expand_pseudo_similar(adj, u=0, v=3, sigma=sigma, target_size=3)

    # Must contain the seed pair
    assert mapping[0] == 3
    # Must be a partial automorphism
    assert is_paut(adj, mapping)
    assert len(mapping) >= 3


def test_bfs_expand_pseudo_similar_limited_by_sigma_coverage() -> None:
    # Star graph: 0 is the center connected to 1, 2, 3
    adj = {0: {1, 2, 3}, 1: {0}, 2: {0}, 3: {0}}
    # sigma only covers node 1 -> leaves 2, 3 unmappable
    sigma = {1: 2}
    mapping = bfs_expand_pseudo_similar(adj, u=0, v=3, sigma=sigma, target_size=3)

    assert mapping[0] == 3
    assert is_paut(adj, mapping)
    # Can grow at most to size 2 ({0: 3, 1: 2}) since sigma has no entry for 2 or 3
    assert len(mapping) <= 2


# --- paut_sizes_to_csv ---


def test_paut_sizes_to_csv(tmp_path: Path) -> None:
    stats = {
        5: [
            PautStats(
                paut_size=3,
                label=1,
                dataset_type=DatasetType.TRAIN,
                strategy="positive",
            )
        ],
        7: [
            PautStats(
                paut_size=4, label=0, dataset_type=DatasetType.VAL, strategy="blocking"
            )
        ],
    }
    csv_path = tmp_path / "stats.csv"
    paut_sizes_to_csv(stats, str(csv_path))

    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "num_of_nodes,paut_size,label,dataset_type,strategy"
    assert "5,3,1,train,positive" in lines[1]
    assert "7,4,0,val,blocking" in lines[2]


# --- construct_pseudo_similar_graph ---
# TODO


def test_construct_pseudo_similar_graph_structure() -> None:
    # Path 0-1-2-3 with automorphism (0 3)(1 2)
    base_adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    sigma = [3, 2, 1, 0]  # (0 3)(1 2) in array form
    S = {0, 1}  # u is adjacent to S; sigma(S) = {3, 2} for v

    adj_G, n, u, v, witness = construct_pseudo_similar_graph(base_adj, 4, sigma, S)

    assert n == 6
    assert u == 4
    assert v == 5
    # u adjacent to S = {0, 1}
    assert adj_G[u] == {0, 1}
    assert u in adj_G[0] and u in adj_G[1]
    # v adjacent to sigma(S) = {3, 2}
    assert adj_G[v] == {2, 3}
    assert v in adj_G[2] and v in adj_G[3]
    # witness maps base nodes via sigma and u -> v
    assert witness[0] == 3
    assert witness[1] == 2
    assert witness[u] == v
    # Original edges preserved
    assert 1 in adj_G[0] and 0 in adj_G[1]


def test_construct_pseudo_similar_graph_witness_is_valid_isomorphism() -> None:
    # Verify that G-v ≅ G-u via the witness (witness maps V(G-v) → V(G-u))
    base_adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    sigma = [3, 2, 1, 0]
    S = {0, 1}

    adj_G, n, u, v, witness = construct_pseudo_similar_graph(base_adj, 4, sigma, S)

    # Collect edges of G-v (remove v and its edges) — this is the witness domain
    edges_minus_v = set()
    for a, neighbors in adj_G.items():
        if a == v:
            continue
        for b in neighbors:
            if b == v:
                continue
            edges_minus_v.add((a, b))

    # Collect edges of G-u (remove u and its edges) — this is the witness codomain
    edges_minus_u = set()
    for a, neighbors in adj_G.items():
        if a == u:
            continue
        for b in neighbors:
            if b == u:
                continue
            edges_minus_u.add((a, b))

    # Mapping edges of G-v through witness should give edges of G-u
    mapped_edges = {(witness[a], witness[b]) for a, b in edges_minus_v}
    assert mapped_edges == edges_minus_u


# --- find_pseudo_similar_construction ---
# TODO


def test_find_pseudo_similar_construction_returns_none_for_large_graph() -> None:
    # Graph with 21 nodes exceeds MAX_CONSTRUCTED_NODES - 2 = 20
    adj = {i: {i + 1} for i in range(20)}
    adj[20] = {19}
    for i in range(1, 20):
        adj[i].add(i - 1)

    from pynauty import Graph, autgrp

    g = Graph(21)
    g.set_adjacency_dict(adj)
    gens = autgrp(g)[0]
    group = PermutationGroup([Permutation(p) for p in gens])

    result = find_pseudo_similar_construction(adj, 21, group)
    assert result is None


def test_find_pseudo_similar_construction_succeeds_on_path_graph(
    path_graph_group: PermutationGroup,
) -> None:
    import random

    random.seed(42)
    adjacency_dict = {
        0: {1},
        1: {0, 2},
        2: {1, 3, 4},
        3: {2, 4},
        4: {2, 3, 5},
        5: {4, 6},
        6: {5},
    }

    result = find_pseudo_similar_construction(adjacency_dict, 7, path_graph_group)

    if result is not None:
        adj_G, n, u, v, witness = result
        assert n == 9
        # u and v must be the two new vertices
        assert u == 7
        assert v == 8
        # Witness must map u -> v
        assert witness[u] == v
        # The constructed graph must have the right number of nodes
        assert len(adj_G) == 9
