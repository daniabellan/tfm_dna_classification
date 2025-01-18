import numpy as np
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import mlflow.models.signature as mfs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

from datasets.synthetic_dataset import SyntheticDataset
from models.hybrid_model import HybridSequenceClassifier

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device



def load_model_from_config(config_path, model_class):
    """
    Carga un modelo basado en los parámetros especificados en un archivo YAML.

    Args:
        config_path (str): Ruta al archivo de configuración YAML.
        model_class (nn.Module): Clase del modelo a instanciar.

    Returns:
        nn.Module: Modelo instanciado.
    """
    # Leer la configuración del archivo YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Instanciar el modelo usando los parámetros de configuración
    device = get_device()
    model = model_class(**config).to(device)

    return model


def create_synthetic_dataset(dataset_config:str):
    # Leer configuración desde el archivo YAML
    dataset = SyntheticDataset(config_file=dataset_config)

    # Dividir el dataset en entrenamiento y validación usando train_test_split
    train_signals, val_signals, train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        dataset.signals, dataset.sequences, dataset.labels, test_size=0.2, random_state=42)

    # Crear datasets de PyTorch para entrenamiento y validación
    train_dataset = SyntheticDataset(config_file=dataset_config)
    train_dataset.signals, train_dataset.sequences, train_dataset.labels = train_signals, train_sequences, train_labels

    val_dataset = SyntheticDataset(config_file=dataset_config)
    val_dataset.signals, val_dataset.sequences, val_dataset.labels = val_signals, val_sequences, val_labels

    # Crear los DataLoader para entrenamiento y validación
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    # Supongamos que HybridSequenceClassifier ya está definido
    config_path = "config/models_config/hybrid_model_default.yaml"  # Ruta al archivo de configuración
    model = load_model_from_config(config_path, HybridSequenceClassifier)

    # Cargar dataset sintético
    dataset_config = "config/datasets_config/synthetic_dataset_default.yaml"
    train_loader, val_loader = create_synthetic_dataset(dataset_config)
        