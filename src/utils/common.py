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

