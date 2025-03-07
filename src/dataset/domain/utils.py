
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from src.dataset.load_dataset import RealDataset
from src.dataset.collator import SequenceSignalsCollator


def load_dataset(config: dict) -> ConcatDataset:
    """
    Loads and concatenates multiple datasets based on provided paths.

    Args:
        config (dict): Configuration dictionary containing dataset paths.

    Returns:
        ConcatDataset: A concatenated PyTorch dataset.
    """
    datasets = [
        RealDataset(config=config, dataset_path=dataset_path, class_idx=class_idx)
        for class_idx, dataset_path in enumerate(config["dataset_paths"])
    ]
    return ConcatDataset(datasets)


def split_dataset(config: dict, full_dataset: ConcatDataset):
    """
    Splits a dataset into training, validation, and test sets.

    Args:
        config (dict): Configuration dictionary containing split ratios.
        full_dataset (ConcatDataset): The full dataset.

    Returns:
        Tuple[torch.utils.data.dataset.Subset, torch.utils.data.dataset.Subset, torch.utils.data.dataset.Subset]:
        - train_split: Training subset.
        - val_split: Validation subset.
        - test_split: Test subset.
    """
    torch_rng = torch.Generator().manual_seed(42)  # Reproducibility

    train_ratio = config["train_ratio"]
    val_ratio = config["val_ratio"]
    test_ratio = config["test_ratio"]

    return random_split(full_dataset, [train_ratio, val_ratio, test_ratio], generator=torch_rng)


def create_dataloaders(config: dict, kmers_size: int, train_split, val_split, test_split):
    """
    Creates PyTorch DataLoaders for training, validation, and testing.

    Args:
        config (dict): Configuration dictionary containing batch size.
        kmers_size (int): Size of k-mers used in SequenceSignalsCollator.
        train_split (Subset): Training dataset split.
        val_split (Subset): Validation dataset split.
        test_split (Subset): Test dataset split.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train, validation, and test DataLoaders.
    """
    batch_size = config.get("batch_size", 16)
    collate_fn = SequenceSignalsCollator(
        kmers_size=kmers_size,
        max_signal_length=config.get("collator_max_signal_length", 1000),
        max_kmers_length=config.get("collator_max_kmers_length", 45000)
    )

    return (
        DataLoader(train_split, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(val_split, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        DataLoader(test_split, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
    )
