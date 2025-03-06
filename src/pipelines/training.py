import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from pathlib import Path
import yaml
import json

from src.dataset.load_dataset import RealDataset
from src.dataset.collator import SequenceSignalsCollator
from src.utils.common import parse_args, load_config

torch.manual_seed(42)

def load_dataset(config: dict) -> ConcatDataset:
    """
    Loads multiple datasets from the specified paths and concatenates them into a single dataset.

    Args:
        config (dict): Configuration dictionary containing dataset paths.

    Returns:
        ConcatDataset: A PyTorch dataset containing all loaded datasets.
    """
    datasets = [
        RealDataset(config=config, dataset_path=dataset_path, class_idx=class_idx)
        for class_idx, dataset_path in enumerate(config["dataset_paths"])
    ]

    return ConcatDataset(datasets)


def split_dataset(config: dict, full_dataset: ConcatDataset):
    """
    Splits a dataset into training, validation, and test sets based on the specified ratios.

    Args:
        config (dict): Configuration dictionary containing train, validation, and test split ratios.
        full_dataset (ConcatDataset): The full dataset to be split.

    Returns:
        Tuple[torch.utils.data.dataset.Subset, torch.utils.data.dataset.Subset, torch.utils.data.dataset.Subset]:
        - train_split: Training subset.
        - val_split: Validation subset.
        - test_split: Test subset.
    """
    train_ratio = config["train_ratio"]
    val_ratio = config["val_ratio"]
    test_ratio = config["test_ratio"]

    torch_rng = torch.Generator().manual_seed(42)  # Set random seed for reproducibility
    train_split, val_split, test_split = random_split(
        full_dataset, [train_ratio, val_ratio, test_ratio], generator=torch_rng
    )

    return train_split, val_split, test_split


def create_dataloaders(
    config: dict,
    kmers_size: int,
    train_split: torch.utils.data.dataset.Subset,
    val_split: torch.utils.data.dataset.Subset,
    test_split: torch.utils.data.dataset.Subset,
):
    """
    Creates PyTorch DataLoaders for training, validation, and testing.

    Args:
        config (dict): Configuration dictionary containing batch size.
        kmers_size (int): Size of k-mers used in SequenceSignalsCollator.
        train_split (Subset): Training dataset split.
        val_split (Subset): Validation dataset split.
        test_split (Subset): Test dataset split.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]:
        - train_loader: DataLoader for training data.
        - val_loader: DataLoader for validation data.
        - test_loader: DataLoader for test data.
    """
    batch_size = config.get("batch_size", 16)
    collate_fn = SequenceSignalsCollator(kmers_size=kmers_size)

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    config_path = args.config

    # Load all configurations from the training experiment file
    config = load_config(config_path)

    # Load and concatenate datasets
    full_dataset = load_dataset(config["dataset"])

    # Split dataset into training, validation, and test sets
    train_split, val_split, test_split = split_dataset(
        config=config["dataset"], full_dataset=full_dataset
    )

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config=config["training"],
        kmers_size=config["dataset"]["kmers_size"],
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
    )

    for batch_idx, (signals, sequences, labels) in enumerate(train_loader):
        print(f"Mini-batch: {batch_idx}")

    pass