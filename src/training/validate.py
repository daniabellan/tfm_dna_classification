import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
        dict: A dictionary containing validation metrics: loss, accuracy, precision, recall, and F1-score.
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

    # Compute evaluation metrics
    metrics = {
        "loss": running_loss / len(data_loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

    # Free up unused GPU memory
    torch.cuda.empty_cache()

    return metrics
