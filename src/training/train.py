import time
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.hybrid_model import HybridSequenceClassifier

def train_one_epoch(model:HybridSequenceClassifier, 
                    data_loader:torch.utils.data.DataLoader, 
                    criterion:torch.nn.modules.loss, 
                    optimizer:torch.optim.Optimizer, 
                    device:torch.device, 
                    max_grad_norm:float=1.0, 
                    log_interval:int=1) -> dict:
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The neural network model to train.
        data_loader (torch.utils.data.DataLoader): DataLoader providing training batches.
        criterion (torch.nn.Module): Loss function used for optimization.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        device (torch.device): Device to perform computations (e.g., 'cuda' or 'cpu').
        max_grad_norm (float, optional): Maximum norm for gradient clipping to prevent exploding gradients. Default is 1.0.
        log_interval (int, optional): Frequency (in batches) of logging training progress. Default is 1.

    Returns:
        dict: A dictionary containing training metrics: loss, accuracy, precision, recall, and F1-score.
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

    # Compute evaluation metrics
    metrics = {
        "loss": running_loss / len(data_loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

    return metrics
