import time as time
import torch
from torch.utils.data import DataLoader

from datasets.synthetic_dataset import SyntheticDataset
from datasets.real_synthetic_dataset import RealSyntheticDataset
from datasets.collate import SequenceSignalsCollator
from models.hybrid_model import HybridSequenceClassifier
from training.mlflow_logger import MLFlowLogger
from training.train import train_one_epoch
from training.validate import validate
from training.test import test_model
from training.callbacks import EarlyStopping
from utils.common import load_experiment_config, get_device, print_epoch_summary


def load_model_from_config(model_config, model_class):
    # Instanciar el modelo usando los parámetros de configuración
    model = model_class(**model_config)
    return model


def create_synthetic_dataset(dataset_config: dict):
    # Cargar dataset reales y sintéticos
    if "real_dataset" in dataset_config: 
        dataset = RealSyntheticDataset(config = dataset_config)
    else:
        # Crear dataset sintético
        dataset = SyntheticDataset(config = dataset_config)
    # Crear datasets de PyTorch para entrenamiento y val
    train_split = dataset_config["dataset"]["train_split"]
    val_split = dataset_config["dataset"]["val_split"]
    test_split = dataset_config["dataset"]["test_split"]

    splt_percent = [train_split, val_split, test_split]

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, splt_percent)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, train_config: dict):
    # Crear los DataLoader para entrenamiento y validación
    train_loader = DataLoader(train_dataset, 
                              batch_size=train_config["batch_size"], 
                              shuffle=True, 
                              collate_fn=SequenceSignalsCollator(vocab=train_dataset.dataset.vocab,
                                                                  padding_idx=train_dataset.dataset.padding_idx))
    val_loader = DataLoader(val_dataset, 
                            batch_size=train_config["batch_size"], 
                            shuffle=False, 
                            collate_fn=SequenceSignalsCollator(vocab=val_dataset.dataset.vocab,
                                                                padding_idx=val_dataset.dataset.padding_idx))

    test_loader = DataLoader(test_dataset, 
                             batch_size=train_config["batch_size"], 
                             shuffle=False, 
                             collate_fn=SequenceSignalsCollator(vocab=test_dataset.dataset.vocab,
                                                                 padding_idx=test_dataset.dataset.padding_idx))

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Cargar configuracion de experimento
    experiment_config = "config/experiments/synthetic_default.yaml"
    experiment_config = "config/experiments/real_synthetic_default.yaml"
    experiment_name, dataset_config, model_config, train_config = load_experiment_config(experiment_config)

    # Definir device
    device = get_device()

    # Cargar dataset sintético
    train_dataset, val_dataset, test_dataset = create_synthetic_dataset(dataset_config)
    
    # Cargar DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, 
                                                               val_dataset, 
                                                               test_dataset,
                                                               train_config)

    # Supongamos que HybridSequenceClassifier ya está definido
    # Inicializamos el valor máximo de canales en 0
    max_channels = 0

    # Iterar a través de los DataLoaders para obtener el máximo número de canales
    for loader in [train_loader, val_loader, test_loader]:
        for signals, sequences, _ in loader:
            max_channels = max(max_channels, signals.shape[1])  # signals.shape[1] es el número de canales
            break  # Solo mirar el primer batch de cada DataLoader

    # Asignar el máximo número de canales a model_config
    model_config["input_channels"] = max_channels
    print(f"El número máximo de canales es: {max_channels}")
    model = load_model_from_config(model_config, HybridSequenceClassifier).to(device)

    # Configuración de MLFlow
    mlflow_logger = MLFlowLogger(experiment_config = train_config,
                                 model_config = model_config,
                                 experiment_name = experiment_name,
                                 len_train_dataset = len(train_dataset),
                                 len_val_dataset = len(val_dataset),
                                 len_test_dataset = len(test_dataset))

    # Iniciar el temporizador
    mlflow_logger.start_timer()

    # Inicialización
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=train_config["learning_rate"], 
                                 weight_decay=train_config["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=train_config["scheduler"]["step_size"], 
                                                gamma=train_config["scheduler"]["gamma"])
    early_stopping = EarlyStopping(train_config["early_stopping"]["patience"], 
                                   train_config["early_stopping"]["min_delta"])

    # Entrenamiento
    start_train = time.time()
    for epoch in range(train_config["epochs"]):
        start_epoch = time.time()
        
        # Temporizador para la fase de entrenamiento
        start_train_phase = time.time()
        train_metrics = train_one_epoch(model, 
                                        train_loader, 
                                        criterion, 
                                        optimizer, 
                                        device, 
                                        train_config["gradient_clipping"]["max_grad_norm"], 
                                        dataset_config["padding_idx"])
        
        train_time = time.time() - start_train_phase
        
        # Temporizador para la fase de validación
        start_val_phase = time.time()
        val_metrics = validate(model, 
                               val_loader, 
                               criterion, 
                               device, 
                               dataset_config["padding_idx"])
        val_time = time.time() - start_val_phase

        # Registrar tiempos en MLflow
        epoch_time = time.time() - start_epoch
        mlflow_logger.log_metrics({"epoch_time": epoch_time}, epoch)

        # Registrar métricas
        mlflow_logger.log_metrics({"train_" + k: v for k, v in train_metrics.items()}, epoch)
        mlflow_logger.log_metrics({"val_" + k: v for k, v in val_metrics.items()}, epoch)

        scheduler.step()

        elapsed_time = time.time() - start_train
        print_epoch_summary(epoch, train_metrics, val_metrics, epoch_time, elapsed_time)

        if early_stopping(val_metrics["loss"]):
            print("Early stopping triggered")
            break

    # Tiempo total de entrenamiento
    total_train_time = time.time() - start_train
    mlflow_logger.log_metrics({"train_time": total_train_time}, epoch)
    print(f"\n** Training done in {total_train_time:.4f} seconds\n")

    # Test
    start_test = time.time()
    test_metrics, confusion_matrix = test_model(model, 
                                                test_loader, 
                                                criterion, 
                                                device, 
                                                dataset_config["padding_idx"],
                                                num_classes=model_config["num_classes"])
    mlflow_logger.log_metrics({f"test_{k}": v for k, v in test_metrics.items()}, epoch)

    # Usar la función modularizada para imprimir las métricas de test
    elapsed_time = time.time() - start_train
    # Print test results
    print(f"\n# ========== Test ==========")
    print(f"Loss: {test_metrics['loss']:.4f} | Accuracy: {test_metrics['accuracy']:.4f} | Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")
    print(f"Test done in {(time.time() - start_test):.4f} seconds\n")

    mlflow_logger.end_mlflow()
