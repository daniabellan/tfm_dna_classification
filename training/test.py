import torch

import torchmetrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def test_model(model, loader, criterion, device, padding_idx, num_classes):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    # Inicializar la métrica de matriz de confusión
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)

    with torch.no_grad():
        for signals, sequences, labels in loader:
            signals, sequences, labels = signals.to(device), sequences.to(device), labels.to(device)

            # Crear el padding mask
            padding_mask = (sequences == padding_idx)

            # Realizar predicciones
            outputs = model(signals, sequences, padding_mask)
            loss = criterion(outputs, labels)

            # Actualizar la matriz de confusión
            preds = torch.argmax(outputs, dim=1)
            confusion_matrix.update(preds, labels)

            # Acumular resultados
            running_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcular las métricas
    metrics = {
        "loss": running_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

    # Obtener la matriz de confusión
    cm = confusion_matrix.compute().cpu().numpy()

    # Guardar la matriz de confusión en un archivo de texto
    # with open("confusion_matrix.txt", "w") as f:
    #     f.write("Confusion Matrix:\n")
    #     f.write(str(cm))

    # Imprimir la matriz de confusión por consola
    print("Confusion Matrix:")
    print(cm)

    return metrics, cm
