import time as time
import torch
from torch.utils.data import DataLoader

from datasets.synthetic_dataset import SyntheticDataset
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
    # Crear dataset sintético
    dataset = SyntheticDataset(config=dataset_config)
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
    model = load_model_from_config(model_config, HybridSequenceClassifier).to(device)

    # Configuración de MLFlow
    mlflow_logger = MLFlowLogger(train_config, experiment_name)

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
        
        # Train
        train_metrics = train_one_epoch(model, 
                                        train_loader, 
                                        criterion, 
                                        optimizer, 
                                        device, 
                                        train_config["gradient_clipping"]["max_grad_norm"], 
                                        dataset_config["padding_idx"])
        
        # Validate
        val_metrics = validate(model, 
                               val_loader, 
                               criterion, 
                               device, 
                               dataset_config["padding_idx"])

        # Registrar los tiempos de la época
        epoch_time = time.time() - start_epoch
        mlflow_logger.log_epoch_time(epoch_time)

        mlflow_logger.log_metrics({"train_" + k: v for k, v in train_metrics.items()}, epoch)
        mlflow_logger.log_metrics({"val_" + k: v for k, v in val_metrics.items()}, epoch)

        scheduler.step()

        epoch_time = time.time() - start_epoch
        elapsed_time = time.time() - start_train
        # Print epoch summary
        print_epoch_summary(epoch, train_metrics, val_metrics, epoch_time, elapsed_time)

        if early_stopping(val_metrics["loss"]):
            print("Early stopping triggered")
            break

    print(f"\n** Training done in {(time.time() - start_train):4f} seconds\n")

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
