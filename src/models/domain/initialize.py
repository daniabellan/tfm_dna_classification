import torch
from src.models.hybrid_model import HybridSequenceClassifier

def initialize_model(config: dict, dataset_config: dict, device: torch.device):
    """
    Initializes the HybridSequenceClassifier model.

    Args:
        config (dict): Model configuration dictionary.
        dataset_config (dict): Dataset configuration (for num_classes, kmers_size).
        device (torch.device): Target device (CPU/GPU).

    Returns:
        torch.nn.Module: Initialized model.
    """
    model = HybridSequenceClassifier(
        **config,
        kmers_size=dataset_config["kmers_size"],
        num_classes=len(dataset_config["dataset_paths"])
    )

    return model.to(device)