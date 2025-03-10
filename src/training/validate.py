import torch
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.logging_config import logger, get_log_file
from src.models.hybrid_model import HybridSequenceClassifier

def validate(model: HybridSequenceClassifier, 
             data_loader: torch.utils.data.DataLoader, 
             criterion: torch.nn.Module, 
             device: torch.device) -> dict:
    """
    Validates the model on a given dataset.

    Args:
        model (HybridSequenceClassifier): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader providing validation batches.
        criterion (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): Device to perform computations (e.g., 'cuda' or 'cpu').

    Returns:
        dict: A dictionary containing validation loss, accuracy, precision, recall, and F1-score.
    """
    
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():  # Disable gradient computation for efficiency
        for signals, sequences, labels in data_loader:
            # Move tensors to the specified device
            signals, sequences, labels = signals.to(device), sequences.to(device), labels.to(device)

            # Forward pass
            outputs = model(signals, sequences)
            loss = criterion(outputs, labels)

            # Accumulate loss
            running_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Compute multi-class metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Aggregate metrics into a dictionary
    metrics = {
        "val_loss": running_loss / len(data_loader),
        "val_accuracy": accuracy,
        "val_precision_macro": precision_macro,
        "val_recall_macro": recall_macro,
        "val_f1_macro": f1_macro
    }

    

    # Log Validation Summary in the Required Format
    print(f"[Val]   Loss: {metrics['val_loss']:.4f} | Accuracy: {metrics['val_accuracy']:.4f} | "
          f"Precision: {metrics['val_precision_macro']:.4f} | Recall: {metrics['val_recall_macro']:.4f} | "
          f"F1: {metrics['val_f1_macro']:.4f}")

    mlflow.log_artifact(str(get_log_file()))
    logger.info(f"[Val]   Loss: {metrics['val_loss']:.4f} | Accuracy: {metrics['val_accuracy']:.4f} | "
          f"Precision: {metrics['val_precision_macro']:.4f} | Recall: {metrics['val_recall_macro']:.4f} | "
          f"F1: {metrics['val_f1_macro']:.4f}")

    # Free up unused GPU memory
    torch.cuda.empty_cache()

    return metrics
