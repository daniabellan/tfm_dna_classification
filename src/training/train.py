import time
import torch
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.hybrid_model import HybridSequenceClassifier
from src.utils.logging_config import logger, get_log_file


def train_one_epoch(model: HybridSequenceClassifier, 
                    data_loader: torch.utils.data.DataLoader, 
                    criterion: torch.nn.modules.loss, 
                    optimizer: torch.optim.Optimizer, 
                    device: torch.device, 
                    max_grad_norm: float = 1.0, 
                    log_interval: int = 1
                    ) -> dict:
    """
    Trains the model for one epoch with multi-class evaluation metrics.

    Args:
        model (torch.nn.Module): The neural network model to train.
        data_loader (torch.utils.data.DataLoader): DataLoader providing training batches.
        criterion (torch.nn.Module): Loss function used for optimization.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        device (torch.device): Device to perform computations (e.g., 'cuda' or 'cpu').
        max_grad_norm (float, optional): Maximum norm for gradient clipping to prevent exploding gradients. Default is 1.0.
        log_interval (int, optional): Frequency (in batches) of logging training progress. Default is 1.
        
    Returns:
        dict: A dictionary containing training loss, accuracy, precision, recall, and F1-score.
    """

    model.train()  # Set the model to training mode
    running_loss = 0.0
    all_preds, all_labels = [], []

    start_time = time.time()

    for batch_idx, (signals, sequences, labels) in enumerate(data_loader):  
        # Move data to the specified device
        signals, sequences, labels = signals.to(device), sequences.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(signals, sequences)
        loss = criterion(outputs, labels)

        # Backward pass and optimization step
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Track loss and predictions
        running_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)  # Get predicted labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Logging
        if batch_idx % log_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"  Batch {batch_idx + 1}/{len(data_loader)} - Loss: {loss.item():.4f}")
            print(f"  Elapsed Time: {elapsed_time:.4f} sec")
            
            logger.info(f"  Batch {batch_idx + 1}/{len(data_loader)} - Loss: {loss.item():.4f}")
            logger.info(f"  Elapsed Time: {elapsed_time:.4f} sec")
        
        mlflow.log_artifact(str(get_log_file()))

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Compute multi-class metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Aggregate all metrics
    metrics = {
        "train_loss": running_loss / len(data_loader),
        "train_accuracy": accuracy,
        "train_precision_macro": precision_macro,
        "train_recall_macro": recall_macro,
        "train_f1_macro": f1_macro
    }

    # Log Training Summary in the Required Format
    print(f"\n[Train] Loss: {metrics['train_loss']:.4f} | Accuracy: {metrics['train_accuracy']:.4f} | "
          f"Precision: {metrics['train_precision_macro']:.4f} | Recall: {metrics['train_recall_macro']:.4f} | "
          f"F1: {metrics['train_f1_macro']:.4f}")

    logger.info(f"[Train] Loss: {metrics['train_loss']:.4f} | Accuracy: {metrics['train_accuracy']:.4f} | "
                f"Precision: {metrics['train_precision_macro']:.4f} | Recall: {metrics['train_recall_macro']:.4f} | "
                f"F1: {metrics['train_f1_macro']:.4f}")
    mlflow.log_artifact(str(get_log_file()))

    return metrics
