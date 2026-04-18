import random
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple, cast

import networkx as nx
import pynauty
from sympy.combinatorics import Permutation, PermutationGroup

type Edge = tuple[int, int]
type Mapping = dict[int, int]
type AdjacencyDict = dict[int, set[int]]
type OrbitMap = dict[int, int]
type Isomorphism = dict[int, int]

MAX_CONSTRUCTED_NODES = 22


class PseudoSimilarGraph(NamedTuple):
    adj: AdjacencyDict
    num_nodes: int
    u: int
    v: int
    witness: Isomorphism


@dataclass
class GraphData:
    graph: pynauty.Graph
    num_of_nodes: int
    adjacency_dict: AdjacencyDict


def build_orbit_map(group: PermutationGroup) -> OrbitMap:
    """Map each node to its orbit index in the automorphism group.

    :param group: The automorphism group of the graph.
    :returns: Dictionary mapping each node to its orbit index (0-based).
    """
    return {node: i for i, orbit in enumerate(group.orbits()) for node in orbit}


def build_adjacency_dict(edge_list: Iterable[Edge]) -> AdjacencyDict:
    """Build an adjacency dictionary from a list of edges.

    :param edge_list: Iterable of edges as (u, v) tuples.
    :returns: Dictionary mapping each node to its set of neighbors.
    """
    adjacency_dict: AdjacencyDict = {}
    for u, v in edge_list:
        adjacency_dict.setdefault(u, set()).add(v)
        adjacency_dict.setdefault(v, set()).add(u)
    return adjacency_dict


def read_graphs_from_g6(file_path: str) -> list[GraphData]:
    """Read graphs from a .g6 file and convert them to pynauty format.

    :param file_path: Path to the .g6 file.
    :returns: List of GraphData objects containing the pynauty graph, number of nodes, and adjacency dictionary.
    """
    graphs: nx.Graph | list[nx.Graph] = nx.read_graph6(file_path)
    if isinstance(graphs, nx.Graph):
        graphs = [graphs]

    graph_data_list: list[GraphData] = []
    for graph in graphs:
        num_of_nodes = graph.number_of_nodes()
        adjacency_dict = build_adjacency_dict(list(graph.edges()))
        pynauty_graph = pynauty.Graph(num_of_nodes)
        pynauty_graph.set_adjacency_dict(adjacency_dict)
        graph_data_list.append(GraphData(pynauty_graph, num_of_nodes, adjacency_dict))
    return graph_data_list


def is_injective(mapping: Mapping) -> bool:
    """Check if the mapping is injective (one-to-one).

    :param mapping: A partial mapping from node indices to node indices.
    :returns: True if the mapping is injective, False otherwise.
    """
    return len(set(mapping.values())) == len(mapping)


def is_paut(adjacency_dict: AdjacencyDict, mapping: Mapping) -> bool:
    """Check if mapping is a partial automorphism on given graph.

    :param adjacency_dict: Adjacency dictionary of the graph.
    :param mapping: A partial mapping from node indices to node indices.
    :returns: True if the mapping is a partial automorphism, False otherwise.
    """
    if not mapping:
        return False

    if not is_injective(mapping):
        return False

    domain = list(mapping.keys())
    for i, u in enumerate(domain):
        for v in domain[i + 1 :]:
            u_mapped = mapping[u]
            v_mapped = mapping[v]
            if (v in adjacency_dict.get(u, set())) != (
                v_mapped in adjacency_dict.get(u_mapped, set())
            ):
                return False
    return True


def is_extensible(group: PermutationGroup, mapping: Mapping) -> bool:
    """Check if mapping can be extended to a full automorphism on given graph.

    :param group: The automorphism group of the graph.
    :param mapping: A partial mapping from node indices to node indices.
        Indices must be contiguous non-negative integers matching the pynauty graph representation.
    :returns: True if the mapping can be extended to a full automorphism, False otherwise.
    """
    if not mapping:
        return True

    # Quick rejection: every src -> dst must have src and dst in the same orbit.
    orbit_of = build_orbit_map(group)
    for src, dst in mapping.items():
        if orbit_of.get(src) != orbit_of.get(dst):
            return False

    # Full check: enumerate group elements.
    domain = set(mapping.keys())
    for perm in group.generate():
        if all(cast(Permutation, perm).array_form[i] == mapping[i] for i in domain):
            return True
    return False


def bfs_expand_pseudo_similar(
    adj: AdjacencyDict,
    u: int,
    v: int,
    sigma: Isomorphism,
    target_size: int,
) -> Mapping:
    """Grow the seed mapping {u: v} outward by BFS.

    For each visited node w, sigma[w] is the candidate image. The node is
    added to the domain only when w → sigma[w] keeps the mapping a valid
    partial automorphism of the *original* graph.

    Because u and v are pseudo-similar (different orbits), any mapping that
    contains u → v is guaranteed non-extendable, regardless of its size.

    :param adj: Adjacency dictionary of the original graph.
    :param u: Seed source vertex.
    :param v: Seed target vertex (image of u).
    :param sigma: Witnessing isomorphism V(G-v) -> V(G-u).
    :param target_size: Stop once the mapping reaches this size.
    :returns: A partial automorphism containing u → v, of size ≤ target_size.
    """
    mapping: Mapping = {u: v}
    used_targets: set[int] = {v}

    queue: deque[int] = deque([u])
    visited: set[int] = {u}

    while queue and len(mapping) < target_size:
        current = queue.popleft()
        neighbors = list(adj.get(current, set()))
        random.shuffle(neighbors)

        for w in neighbors:
            if w in visited:
                continue
            visited.add(w)

            candidate = sigma.get(w)
            if candidate is None or candidate in used_targets:
                continue

            test_map = {**mapping, w: candidate}
            if is_paut(adj, test_map):
                mapping = test_map
                used_targets.add(candidate)
                queue.append(w)

            if len(mapping) >= target_size:
                break

    return mapping


def construct_pseudo_similar_graph(
    base_adj: AdjacencyDict,
    num_nodes: int,
    sigma: list[int],
    S: set[int],
) -> PseudoSimilarGraph:
    """Godsil-Kocay two-vertex attachment: build G = F + {u,v} with G-u ≅ G-v.

    Given base graph F with automorphism sigma and subset S ⊆ V(F),
    vertex u is adjacent to S and vertex v is adjacent to sigma(S).
    The witnessing isomorphism G-v → G-u is sigma on V(F), u → v.

    :param base_adj: Adjacency dictionary of the base graph F.
    :param num_nodes: Number of nodes in F.
    :param sigma: Automorphism of F in array form [sigma(0), sigma(1), ...].
    :param S: Subset of V(F) — the neighborhood of the new vertex u.
    :returns: (adj_G, |V(G)|, u, v, witness_isomorphism).
    """
    u = num_nodes
    v = num_nodes + 1

    adj: AdjacencyDict = {node: set(nbrs) for node, nbrs in base_adj.items()}
    adj[u] = set()
    adj[v] = set()

    for w in S:
        adj[u].add(w)
        adj.setdefault(w, set()).add(u)

    sigma_S = {sigma[w] for w in S}
    for w in sigma_S:
        adj[v].add(w)
        adj.setdefault(w, set()).add(v)

    witness: Isomorphism = {w: sigma[w] for w in range(num_nodes)}
    witness[u] = v

    return PseudoSimilarGraph(adj, num_nodes + 2, u, v, witness)


def find_pseudo_similar_construction(
    base_adj: AdjacencyDict,
    num_nodes: int,
    group: PermutationGroup,
    max_attempts: int = 200,
) -> PseudoSimilarGraph | None:
    """Search for a Godsil-Kocay construction that yields pseudo-similar vertices.

    Tries random non-identity automorphisms and random subsets S until
    a construction produces vertices u, v in different orbits of G.
    Skips base graphs with more than MAX_CONSTRUCTED_NODES - 2 nodes so
    that the result stays within the dataset's node limit.

    :param base_adj: Adjacency dictionary of the base graph F.
    :param num_nodes: Number of nodes in F.
    :param group: Automorphism group of F.
    :param max_attempts: Maximum random trials.
    :returns: PseudoSimilarGraph or None.
    """
    if num_nodes > MAX_CONSTRUCTED_NODES - 2:
        return None

    elements = [cast(Permutation, phi).array_form for phi in group.generate()]

    for _ in range(max_attempts):
        sigma = random.choice(elements)
        if all(i == sigma[i] for i in range(num_nodes)):
            continue

        k = random.randint(1, max(1, num_nodes - 1))
        S = set(random.sample(range(num_nodes), k))
        S_frozen = frozenset(S)
        sigma_S = frozenset(sigma[w] for w in S)

        if S_frozen == sigma_S:
            continue

        # u and v land in the same orbit iff some φ ∈ Aut(F) swaps S and σ(S).
        # Check this directly on F instead of running nauty on the larger G.
        swapped = any(
            frozenset(phi[w] for w in S_frozen) == sigma_S
            and frozenset(phi[w] for w in sigma_S) == S_frozen
            for phi in elements
        )
        if swapped:
            continue

        return construct_pseudo_similar_graph(base_adj, num_nodes, sigma, S)

    return None
