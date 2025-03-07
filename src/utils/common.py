import argparse

import yaml
from omegaconf import OmegaConf

def load_config(config_path:str):
    """
    Loads the main configuration and all referenced configurations from the file.

    Args:
        config_path (str): Path to the main configuration file.

    Returns:
        OmegaConf.DictConfig: Fully merged configuration.
    """
    # Load the main configuration file
    with open(config_path, "r") as f:
        main_config = yaml.safe_load(f)

    # Convert to an OmegaConf DictConfig object
    config = OmegaConf.create(main_config)

    # List of keys that may contain paths to additional configurations
    config_keys = ["dataset", "training", "model", "logging"]

    # Load and merge each referenced configuration
    for key in config_keys:
        if key in config and isinstance(config[key], str):  # If it's a file path
            with open(config[key], "r") as f:
                sub_config = yaml.safe_load(f)  # Load the sub-configuration
            sub_config = OmegaConf.create(sub_config)  # Convert to DictConfig
            config[key] = sub_config  # Replace the file path with the actual content

    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Train a DNA species classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file (train/test)")
    return parser.parse_args()

def print_epoch_summary(epoch: int, 
                        train_metrics: dict, 
                        val_metrics: dict, 
                        epoch_time: float, 
                        elapsed_time: float, 
                        total_epochs: int = 200):
    """
    Prints a summary of training and validation metrics for a given epoch.

    Args:
        epoch (int): The current epoch number.
        train_metrics (dict): Dictionary containing training metrics (loss, accuracy, precision, recall, F1-score).
        val_metrics (dict): Dictionary containing validation metrics (loss, accuracy, precision, recall, F1-score).
        epoch_time (float): Time taken to complete the current epoch (in seconds).
        elapsed_time (float): Total elapsed training time since the start (in seconds).
        total_epochs (int, optional): Total number of epochs. Defaults to 100.

    Returns:
        None
    """

    print(f"\n# ========== Train Epoch {epoch+1}/{total_epochs} ==========")
    print(f"[Train] Loss: {train_metrics.get('loss', 0):.4f} | Accuracy: {train_metrics.get('accuracy', 0):.4f} "
          f"| Precision: {train_metrics.get('precision', 0):.4f} | Recall: {train_metrics.get('recall', 0):.4f} "
          f"| F1: {train_metrics.get('f1', 0):.4f}")
    
    print(f"[Val]   Loss: {val_metrics.get('loss', 0):.4f} | Accuracy: {val_metrics.get('accuracy', 0):.4f} "
          f"| Precision: {val_metrics.get('precision', 0):.4f} | Recall: {val_metrics.get('recall', 0):.4f} "
          f"| F1: {val_metrics.get('f1', 0):.4f}")
    
    print(f"Epoch Time: {epoch_time:.4f} seconds | Elapsed Time: {elapsed_time:.4f} seconds")

