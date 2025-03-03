import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def validate(model, loader, criterion, device, padding_idx):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for signals, sequences, labels in loader:
            signals, sequences, labels = signals.to(device), sequences.to(device), labels.to(device)

            # Crear el padding mask
            padding_mask = (sequences == padding_idx)

            outputs = model(signals, sequences, padding_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "loss": running_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

    torch.cuda.empty_cache()

    return metrics
