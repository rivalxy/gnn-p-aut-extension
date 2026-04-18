import math
import random
from typing import cast

from sympy.combinatorics import Permutation, PermutationGroup

from dataset.graph_utils import (
    AdjacencyDict,
    Isomorphism,
    Mapping,
    bfs_expand_pseudo_similar,
    build_orbit_map,
    is_extensible,
    is_paut,
)

MAX_ATTEMPTS = 100
MIN_PARTIAL_AUT_FRACTION = 0.5
MAX_PARTIAL_AUT_FRACTION = 0.8
MAX_BLOCKING_CANDIDATES = 5


def is_identity_permutation(perm: list[int]) -> bool:
    """Check if a permutation in array form is the identity.

    :param perm: Permutation in array form [p(0), p(1), ...].
    :returns: True if p(i) == i for all i.
    """
    return all(i == mapped for i, mapped in enumerate(perm))


def mapping_key(mapping: Mapping) -> frozenset[tuple[int, int]]:
    """Convert a mapping to a frozenset of (src, dst) pairs for deduplication.

    :param mapping: A partial automorphism mapping.
    :returns: Frozenset of (source, target) pairs.
    """
    return frozenset(mapping.items())


def partial_size_bounds(
    num_of_nodes: int, upper_bound: int | None = None
) -> tuple[int, int]:
    """Return (min_size, max_size) for sampling a partial automorphism domain.

    :param num_of_nodes: Number of nodes in the graph.
    :param upper_bound: Hard cap on max_size; if None, uses MAX_PARTIAL_AUT_FRACTION.
    :returns: Tuple (min_size, max_size).
    """
    min_size = math.ceil(num_of_nodes * MIN_PARTIAL_AUT_FRACTION)
    max_size = (
        upper_bound
        if upper_bound is not None
        else math.floor(num_of_nodes * MAX_PARTIAL_AUT_FRACTION)
    )
    return min_size, max_size


def sample_partial_size(num_of_nodes: int, upper_bound: int | None = None) -> int:
    """Sample a random partial automorphism domain size within bounds.

    :param num_of_nodes: Number of nodes in the graph.
    :param upper_bound: Hard cap on the maximum size; if None, uses MAX_PARTIAL_AUT_FRACTION.
    :returns: Random integer in [min_size, max_size].
    """
    min_size, max_size = partial_size_bounds(num_of_nodes, upper_bound)
    return random.randint(min_size, max_size)


def is_non_extensible_paut(
    adjacency_list: AdjacencyDict,
    group: PermutationGroup,
    mapping: Mapping,
) -> bool:
    """Check if mapping is a valid partial automorphism that cannot be extended.

    :param adjacency_list: Adjacency dictionary of the graph.
    :param group: Automorphism group of the graph.
    :param mapping: Candidate mapping.
    :returns: True if the mapping is a partial automorphism and not extensible.
    """
    return is_paut(adjacency_list, mapping) and not is_extensible(group, mapping)


def gen_positive_examples(
    group: PermutationGroup, num_of_nodes: int, examples_num: int
) -> list[tuple[Mapping, int]]:
    """Generate extensible partial automorphisms by restricting full automorphisms.

    Samples random non-identity permutations from the group and restricts each
    to a random subset of nodes of size in [MIN_PARTIAL_AUT_FRACTION, MAX_PARTIAL_AUT_FRACTION].

    :param group: Automorphism group of the graph.
    :param num_of_nodes: Number of nodes in the graph.
    :param examples_num: Target number of distinct examples to generate.
    :returns: List of (mapping, paut_size) tuples with label 1 (extensible).
    """
    seen_positives: set[frozenset[tuple[int, int]]] = set()
    positives: list[tuple[Mapping, int]] = []
    attempts = 0
    nodes = list(range(num_of_nodes))

    while len(positives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
        attempts += 1
        perm = cast(Permutation, group.random()).array_form
        if is_identity_permutation(perm):
            continue

        p_aut_size = sample_partial_size(num_of_nodes)
        domain = random.sample(nodes, p_aut_size)
        mapping = {i: perm[i] for i in domain}

        key = mapping_key(mapping)
        if key in seen_positives:
            continue

        seen_positives.add(key)
        positives.append((mapping, len(mapping)))
    return positives


def gen_pseudo_similar_examples(
    group: PermutationGroup,
    num_of_nodes: int,
    adjacency_list: AdjacencyDict,
    u: int,
    v: int,
    witness: Isomorphism,
    examples_num: int,
) -> list[tuple[Mapping, int]]:
    """Generate hard negatives from a Godsil-Kocay constructed pseudo-similar pair.

    The pair (u, v) is guaranteed pseudo-similar and the witness isomorphism
    maps G-v -> G-u, so we grow partial automorphisms containing u -> v
    (which is non-extensible by construction).

    :param group: Automorphism group of the constructed graph G.
    :param num_of_nodes: Number of nodes in G.
    :param adjacency_list: Adjacency dictionary of G.
    :param u: Source pseudo-similar vertex.
    :param v: Target pseudo-similar vertex (different orbit from u).
    :param witness: Isomorphism G-v -> G-u.
    :param examples_num: Maximum number of examples to generate.
    :returns: List of (mapping, paut_size) tuples.
    """
    negatives: list[tuple[Mapping, int]] = []
    seen: set[frozenset[tuple[int, int]]] = set()

    min_size, max_size = partial_size_bounds(num_of_nodes)
    attempts = 0

    while len(negatives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
        attempts += 1
        target_size = random.randint(min_size, max_size)

        mapping = bfs_expand_pseudo_similar(adjacency_list, u, v, witness, target_size)

        if len(mapping) < min_size:
            continue
        if not is_non_extensible_paut(adjacency_list, group, mapping):
            continue

        key = mapping_key(mapping)
        if key in seen:
            continue

        seen.add(key)
        negatives.append((mapping, len(mapping)))

    return negatives


def block_automorphism(
    positive: Mapping, num_of_nodes: int, adj: AdjacencyDict, group: PermutationGroup
) -> Mapping | None:
    """Extend a positive mapping by one cross-orbit assignment to make it non-extensible.

    Tries to find an unmapped node and a target in a different orbit such that
    the extended mapping is still a valid partial automorphism but can no longer
    be extended to a full automorphism.

    :param positive: An extensible partial automorphism mapping.
    :param num_of_nodes: Number of nodes in the graph.
    :param adj: Adjacency dictionary of the graph.
    :param group: Automorphism group of the graph.
    :returns: Extended non-extensible mapping, or None if no such extension is found.
    """
    nodes = list(range(num_of_nodes))
    orbit_of = build_orbit_map(group)

    unmapped_nodes = [n for n in nodes if n not in positive]
    targets = [n for n in nodes if n not in positive.values()]

    if not unmapped_nodes or not targets:
        return None

    random.shuffle(unmapped_nodes)
    random.shuffle(targets)

    for node in unmapped_nodes[: min(MAX_BLOCKING_CANDIDATES, len(unmapped_nodes))]:
        non_orbit_targets = [
            target for target in targets if orbit_of.get(target) != orbit_of.get(node)
        ]

        for target in non_orbit_targets[
            : min(MAX_BLOCKING_CANDIDATES, len(non_orbit_targets))
        ]:
            test_map = positive.copy()
            test_map[node] = target

            if is_non_extensible_paut(adj, group, test_map):
                return test_map

    return None


def gen_blocking_examples(
    group: PermutationGroup,
    examples_num: int,
    num_of_nodes: int,
    adjacency_list: AdjacencyDict,
) -> list[tuple[Mapping, int]]:
    """Generate non-extensible partial automorphisms using the blocking strategy.

    Starts from a random restriction of a full automorphism, then iteratively
    extends it with cross-orbit assignments via block_automorphism until the
    mapping becomes non-extensible.

    :param group: Automorphism group of the graph.
    :param examples_num: Target number of distinct examples to generate.
    :param num_of_nodes: Number of nodes in the graph.
    :param adjacency_list: Adjacency dictionary of the graph.
    :returns: List of (mapping, paut_size) tuples with label 0 (non-extensible).
    """
    negatives: list[tuple[Mapping, int]] = []
    seen_negatives: set[frozenset[tuple[int, int]]] = set()
    attempts = 0
    nodes = list(range(num_of_nodes))
    maximum_size = math.floor(num_of_nodes * MAX_PARTIAL_AUT_FRACTION)

    while len(negatives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
        attempts += 1
        perm = cast(Permutation, group.random()).array_form
        if is_identity_permutation(perm):
            continue

        p_aut_size = sample_partial_size(num_of_nodes, upper_bound=maximum_size) - 1
        domain = random.sample(nodes, p_aut_size)
        mapping = {i: perm[i] for i in domain}

        maximum_extension = maximum_size - p_aut_size
        extension_size = random.randint(1, maximum_extension)
        current_extension = 0
        extension_attempts = 0

        while current_extension < extension_size and extension_attempts < MAX_ATTEMPTS:
            extension_attempts += 1
            new_mapping = block_automorphism(
                mapping, num_of_nodes, adjacency_list, group
            )
            if new_mapping is None:
                break

            mapping = new_mapping
            current_extension += 1

        if not is_non_extensible_paut(adjacency_list, group, mapping):
            continue

        key = mapping_key(mapping)
        if key in seen_negatives:
            continue

        seen_negatives.add(key)
        negatives.append((mapping, len(mapping)))

    return negatives
