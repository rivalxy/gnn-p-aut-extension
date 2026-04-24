import json
import os
import random
from collections import defaultdict

import torch
from sklearn.model_selection import train_test_split

from dataset.build import (
    DatasetConfiguration,
    DatasetType,
    PautStats,
    generate_raw_examples,
    paut_sizes_to_csv,
    raw_examples_to_pyg,
)
from dataset.graph_utils import read_graphs_from_g6


def main() -> None:
    random.seed(42)
    positive_graphs = read_graphs_from_g6("dataset/all_graphs.g6")

    graphs_train, graphs_temp, idx_train, idx_temp = train_test_split(
        positive_graphs, range(len(positive_graphs)), test_size=0.2, random_state=42
    )
    graphs_val, graphs_test, idx_val, idx_test = train_test_split(
        graphs_temp, idx_temp, test_size=0.5, random_state=42
    )

    splits = {
        "train": list(idx_train),
        "val": list(idx_val),
        "test": list(idx_test),
    }
    with open("dataset/splits.json", "w") as f:
        json.dump(splits, f)

    # Shared across all splits: maps nauty canonical certificate of each constructed
    # graph G to the split that first registered it. Val/test are generated first so
    # their G's are registered before train, ensuring train never reuses them.
    seen_canonical: dict[bytes, DatasetType] = {}

    val_test_max_examples = 10
    print("Generating shared val examples...")
    raw_val = generate_raw_examples(
        graphs_val, DatasetType.VAL, val_test_max_examples, seen_canonical
    )
    print(f"Generated {len(raw_val)} raw val examples.")

    print("Generating shared test examples...")
    raw_test = generate_raw_examples(
        graphs_test, DatasetType.TEST, val_test_max_examples, seen_canonical
    )
    print(f"Generated {len(raw_test)} raw test examples.")

    print("Encoding val/test (baseline)...")
    val_dataset_baseline, val_paut_sizes_baseline = raw_examples_to_pyg(
        raw_val, extra_features=False
    )
    test_dataset_baseline, _ = raw_examples_to_pyg(raw_test, extra_features=False)

    print("Encoding val/test (7_features)...")
    val_dataset_7f, val_paut_sizes_7f = raw_examples_to_pyg(
        raw_val, extra_features=True
    )
    test_dataset_7f, _ = raw_examples_to_pyg(raw_test, extra_features=True)

    print("Generating train examples (max_examples=10)...")
    raw_train_10 = generate_raw_examples(graphs_train, DatasetType.TRAIN, 10, seen_canonical)
    print(f"  train: {len(raw_train_10)}")

    print("Generating train examples (max_examples=20)...")
    raw_train_20 = generate_raw_examples(graphs_train, DatasetType.TRAIN, 20, seen_canonical)
    print(f"  train: {len(raw_train_20)}")

    os.makedirs("dataset/baseline", exist_ok=True)
    os.makedirs("dataset/7_features", exist_ok=True)

    torch.save(val_dataset_baseline, "dataset/val_dataset.pt")
    torch.save(test_dataset_baseline, "dataset/test_dataset.pt")

    torch.save(val_dataset_7f, "dataset/7_features/val_dataset_7_features.pt")
    torch.save(test_dataset_7f, "dataset/7_features/test_dataset_7_features.pt")

    configurations = [
        DatasetConfiguration(
            name="baseline",
            raw_train=raw_train_10,
            extra_features=False,
            val_paut_sizes=val_paut_sizes_baseline,
            train_output_path="dataset/baseline/train_dataset_baseline.pt",
            paut_sizes_output_path="dataset/baseline/paut_sizes_baseline.csv",
            val_dataset=val_dataset_baseline,
            test_dataset=test_dataset_baseline,
        ),
        DatasetConfiguration(
            name="7_features",
            raw_train=raw_train_10,
            extra_features=True,
            val_paut_sizes=val_paut_sizes_7f,
            train_output_path="dataset/7_features/train_dataset_7_features.pt",
            paut_sizes_output_path="dataset/7_features/paut_sizes_7_features.csv",
            val_dataset=val_dataset_7f,
            test_dataset=test_dataset_7f,
        ),
        DatasetConfiguration(
            name="larger",
            raw_train=raw_train_20,
            extra_features=False,
            val_paut_sizes=val_paut_sizes_baseline,
            train_output_path="dataset/train_dataset.pt",
            paut_sizes_output_path="dataset/paut_sizes.csv",
            val_dataset=val_dataset_baseline,
            test_dataset=test_dataset_baseline,
        ),
    ]

    for config in configurations:
        print(f"Encoding train dataset for configuration: {config.name}")

        train_dataset, train_paut_sizes = raw_examples_to_pyg(
            config.raw_train, config.extra_features
        )

        paut_sizes: dict[int, list[PautStats]] = defaultdict(list)
        for node_count, stats in train_paut_sizes.items():
            paut_sizes[node_count].extend(stats)
        for node_count, stats in config.val_paut_sizes.items():
            paut_sizes[node_count].extend(stats)

        torch.save(train_dataset, config.train_output_path)
        paut_sizes_to_csv(paut_sizes, config.paut_sizes_output_path)

        print(
            f"Generated {len(train_dataset)} train, {len(config.val_dataset)} val, {len(config.test_dataset)} test examples."
        )


if __name__ == "__main__":
    main()
