import torch

from dataset.build import (
    DatasetType,
    PautStats,
    RawPautExample,
    build_edge_index,
    raw_examples_to_pyg,
)


def test_build_edge_index_empty_graph() -> None:
    edge_index = build_edge_index({})
    assert edge_index.shape == (2, 0)


def test_build_edge_index_non_empty_graph() -> None:
    adjacency_dict = {0: {1, 2}, 1: {0}, 2: {0}}
    edge_index = build_edge_index(adjacency_dict)
    assert edge_index.shape[0] == 2
    # 0-1 and 0-2 each stored in both directions = 4 edges
    assert edge_index.shape[1] == 4
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    assert (0, 1) in edges
    assert (1, 0) in edges
    assert (0, 2) in edges
    assert (2, 0) in edges


def test_raw_examples_to_pyg_converts_and_aggregates_stats() -> None:
    raw_examples = [
        RawPautExample(
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            num_of_nodes=2,
            mapping={0: 1},
            label=1,
            paut_stats=PautStats(
                paut_size=1,
                label=1,
                dataset_type=DatasetType.TRAIN,
                strategy="positive",
            ),
        )
    ]

    pyg_data, paut_sizes = raw_examples_to_pyg(raw_examples, extra_features=False)

    assert len(pyg_data) == 1
    assert pyg_data[0].num_nodes == 2
    assert pyg_data[0].y == 1.0
    assert 2 in paut_sizes
    assert len(paut_sizes[2]) == 1
    assert paut_sizes[2][0].paut_size == 1


def test_raw_examples_to_pyg_with_extra_features() -> None:
    raw_examples = [
        RawPautExample(
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            num_of_nodes=2,
            mapping={0: 1},
            label=0,
            paut_stats=PautStats(
                paut_size=1,
                label=0,
                dataset_type=DatasetType.TRAIN,
                strategy="blocking",
            ),
        )
    ]

    pyg_data, _ = raw_examples_to_pyg(raw_examples, extra_features=True)

    assert len(pyg_data) == 1
    assert pyg_data[0].x is not None
    # Extra features variant has 7 feature columns
    assert pyg_data[0].x.shape == (2, 7)
    assert pyg_data[0].y == 0.0
