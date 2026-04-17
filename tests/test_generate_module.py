from pathlib import Path
from typing import Any

import torch
from pytest import MonkeyPatch
from torch_geometric.data import Data

from dataset.build import DatasetType, PautStats
from dataset.generate import main


def test_main_smoke_with_mocked_io(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_read_graphs_from_g6(_path: str) -> list[int]:
        return list(range(10))

    def fake_generate_raw_examples(*args, **kwargs) -> list[str]:
        return ["raw"]

    def fake_raw_examples_to_pyg(
        raw_examples: list[Any], extra_features: bool
    ) -> tuple[list[Data], dict[int, list[PautStats]]]:
        _ = raw_examples
        feature_count = 7 if extra_features else 3
        dataset = [
            Data(
                x=torch.zeros((1, feature_count), dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                y=torch.tensor([1.0]),
            )
        ]
        stats = {1: [PautStats(1, 1, DatasetType.TRAIN, "positive")]}
        return dataset, stats

    saved_paths: list[str] = []

    def fake_torch_save(_obj, path: str) -> None:
        saved_paths.append(path)

    csv_paths: list[str] = []

    def fake_paut_sizes_to_csv(_stats, path: str) -> None:
        csv_paths.append(path)

    monkeypatch.setattr(
        "dataset.generate.read_graphs_from_g6", fake_read_graphs_from_g6
    )
    monkeypatch.setattr(
        "dataset.generate.generate_raw_examples", fake_generate_raw_examples
    )
    monkeypatch.setattr(
        "dataset.generate.raw_examples_to_pyg", fake_raw_examples_to_pyg
    )
    monkeypatch.setattr("dataset.generate.paut_sizes_to_csv", fake_paut_sizes_to_csv)
    monkeypatch.setattr("dataset.generate.torch.save", fake_torch_save)

    (tmp_path / "dataset").mkdir()
    main()

    assert (tmp_path / "dataset" / "splits.json").exists()
    assert len(saved_paths) == 7
    assert "dataset/val_dataset.pt" in saved_paths
    assert "dataset/test_dataset.pt" in saved_paths
    assert "dataset/7_features/val_dataset_7_features.pt" in saved_paths
    assert "dataset/7_features/test_dataset_7_features.pt" in saved_paths
    assert "dataset/baseline/train_dataset_baseline.pt" in saved_paths
    assert "dataset/7_features/train_dataset_7_features.pt" in saved_paths
    assert "dataset/train_dataset.pt" in saved_paths

    assert sorted(csv_paths) == sorted(
        [
            "dataset/baseline/paut_sizes_baseline.csv",
            "dataset/7_features/paut_sizes_7_features.csv",
            "dataset/paut_sizes.csv",
        ]
    )
