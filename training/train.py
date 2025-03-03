import time
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm, padding_idx, verbose=True):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    start = time.time()
    for batch_idx, (signals, sequences, labels) in enumerate(loader):
        
        signals, sequences, labels = signals.to(device), sequences.to(device), labels.to(device)

        # Crear el padding mask
        padding_mask = (sequences == padding_idx)

        optimizer.zero_grad()
        outputs = model(signals, sequences, padding_mask)

        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Verbose: Imprimir información de cada mini-batch
        if verbose and (batch_idx % 1 == 0):
            print(f"  Mini-Batch {batch_idx + 1}/{len(loader)} - Loss: {loss.item():.4f}")
            print(f"  Elapsed time : {(time.time()-start):.4f}")

    metrics = {
        "loss": running_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

    return metrics
