import time as time
import argparse
import torch

import json
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from datasets.synthetic_dataset import SyntheticDataset
from datasets.real_synthetic_dataset import RealSyntheticDataset, StratifiedDataset
from datasets.collate import SequenceSignalsCollator
from models.hybrid_model import HybridSequenceClassifier
from training.mlflow_logger import MLFlowLogger
from training.train import train_one_epoch
from training.validate import validate
from training.test import test_model
from training.callbacks import EarlyStopping, BestModelCheckpoint
from utils.common import load_experiment_config, get_device, print_epoch_summary

def parse_args():
    parser = argparse.ArgumentParser(description="Train a hybrid sequence classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    return parser.parse_args()


def load_model_from_config(model_config, model_class):
    # Instanciar el modelo usando los parámetros de configuración
    model = model_class(**model_config)
    return model


def create_synthetic_dataset(dataset_config: dict):
    # Cargar dataset reales y sintéticos
    start = time.time()
    gen_dataset = RealSyntheticDataset(config = dataset_config)
    print(f"Data loaded in {(time.time() - start):.4f} seconds")

    # Split estratificado para mantener el balance de las clases entre cada split
    train_dataset = StratifiedDataset(
        gen_dataset=gen_dataset,
        config = dataset_config,
        mode='train'  # Establecer el modo como 'train'
    )

    val_dataset = StratifiedDataset(
        gen_dataset=gen_dataset,
        config = dataset_config,
        mode='val'  # Establecer el modo como 'val'
    )

    test_dataset = StratifiedDataset(
        gen_dataset=gen_dataset,
        config = dataset_config,
        mode='test'  # Establecer el modo como 'test'
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, 
                       val_dataset, 
                       test_dataset, 
                       train_config: dict, 
                       dataset_config:dict):
    
    # Crear los DataLoader para entrenamiento y validación
    train_loader = DataLoader(train_dataset, 
                              batch_size=train_config["batch_size"], 
                              shuffle=True, 
                              collate_fn=SequenceSignalsCollator(padding_idx=dataset_config["padding_idx"]))
    val_loader = DataLoader(val_dataset, 
                            batch_size=train_config["batch_size"], 
                            shuffle=False, 
                            collate_fn=SequenceSignalsCollator(padding_idx=dataset_config["padding_idx"]))

    test_loader = DataLoader(test_dataset, 
                             batch_size=train_config["batch_size"], 
                             shuffle=False, 
                            collate_fn=SequenceSignalsCollator(padding_idx=dataset_config["padding_idx"]))

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    experiment_config_file = args.config

    # Definir configuración del experimento
    try:
        # Cargar experimento
        experiment_config = yaml.safe_load(open(experiment_config_file, "r"))
        experiment_name = Path(args.config).name.strip(".yaml")

        print(f"Running experiment: {experiment_name}")
        print(f"Experiment config: {json.dumps(experiment_config, indent=4)}")

    except:
        raise FileNotFoundError 

    # Cargar configuuracion de entrenamiento
    train_config = experiment_config["train"]

    # Cargar configuración del dataset 
    dataset_config = experiment_config["dataset"]
    # Incluir padding idx para indicar el int que representa el padding de los kmer 
    dataset_config["padding_idx"] = (4**dataset_config["k_mers_size"])

    # Cargar configuracion del modelo
    model_config = experiment_config["model"]
    # Numero de indices que representan los kmers (4^N)
    model_config["vocab_size"] = (4**dataset_config["k_mers_size"])+1

    # Cargar dataset sintético
    train_dataset, val_dataset, test_dataset = create_synthetic_dataset(dataset_config)
    
    # Cargar DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, 
                                                               val_dataset, 
                                                               test_dataset,
                                                               train_config,
                                                               dataset_config)
    # Definir device
    device = get_device()
    
    # Cargar modelo
    model = load_model_from_config(model_config, HybridSequenceClassifier).to(device)

    # Load model (TODO: Not implemented)
    # model_checkpoint = "checkpoints/run_2025-01-31T18-53-05__best_model_epoch_0.pt"
    # model = torch.load(model_checkpoint, weights_only=False)

    # Configuración de MLFlow
    mlflow_logger = MLFlowLogger(train_config = train_config,
                                 model_config = model_config,
                                 experiment_name = experiment_name,
                                 dataset_config = dataset_config,
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

    # Directorio para guardar los checkpoints
    checkpoint_dir = "checkpoints"
    best_checkpoint = BestModelCheckpoint(save_dir=checkpoint_dir,
                                           monitor="loss", 
                                           mode="min",
                                           run_name=mlflow_logger.run_name)


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

        # Guardar el mejor modelo
        best_checkpoint(model, 
                        epoch, 
                        val_metrics)

        # Registrar tiempos en MLflow
        epoch_time = time.time() - start_epoch
        mlflow_logger.log_metrics({"epoch_time": epoch_time}, epoch)

        # Registrar métricas
        mlflow_logger.log_metrics({"train_" + k: v for k, v in train_metrics.items()}, epoch)
        mlflow_logger.log_metrics({"val_" + k: v for k, v in val_metrics.items()}, epoch)

        scheduler.step()

        elapsed_time = time.time() - start_train
        print_epoch_summary(epoch, train_metrics, val_metrics, epoch_time, elapsed_time)

        if early_stopping(val_metrics["accuracy"]):
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
