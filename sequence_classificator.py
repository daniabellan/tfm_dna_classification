import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
import mlflow.models.signature as mfs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# 1. Generación de datos sintéticos con longitud variable
class SyntheticSequenceDataset(Dataset):
    def __init__(self, num_samples, min_seq_length, max_seq_length):
        self.num_samples = num_samples
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        base_probs = [
            [0.4, 0.3, 0.2, 0.1],  # Patrón para la especie 0
            [0.1, 0.4, 0.4, 0.1],  # Patrón para la especie 1
            [0.3, 0.3, 0.2, 0.2],  # Patrón para la especie 2
            [0.2, 0.2, 0.3, 0.3],  # Patrón para la especie 3
            [0.25, 0.25, 0.25, 0.25]  # Patrón para la especie 4 (uniforme)
        ]
        bases = ['A', 'C', 'T', 'G']
        data, labels = [], []

        for label, probs in enumerate(base_probs):
            for _ in range(self.num_samples // len(base_probs)):
                seq_length = np.random.randint(self.min_seq_length, self.max_seq_length + 1)
                sequence = ''.join(np.random.choice(bases, seq_length, p=probs))
                data.append(sequence)
                labels.append(label)

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        return sequence, label

# 2. Convertir secuencias a índices y aplicar padding
class SequenceCollator:
    def __init__(self, vocab, padding_idx=4):
        self.vocab = vocab
        self.padding_idx = padding_idx

    def __call__(self, batch):
        sequences, labels = zip(*batch)

        # Encontrar la longitud máxima de secuencia en el batch
        max_len = max(len(seq) for seq in sequences)
        
        # Codificar las secuencias y hacer padding
        sequences_idx = []
        for seq in sequences:
            seq_idx = [self.vocab[base] for base in seq]
            # Aplicar padding a las secuencias para igualar las longitudes
            padded_seq = seq_idx + [self.padding_idx] * (max_len - len(seq))
            sequences_idx.append(padded_seq)

        return torch.tensor(sequences_idx, dtype=torch.long), torch.tensor(labels, dtype=torch.long), max_len

# 3. Modelo basado en embeddings y transformers (modificado)
class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes, num_layers, max_seq_length):
        super(SequenceClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(0.5)  # Añadido para regularización

    def forward(self, x, padding_mask=None):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = x.mean(dim=1)  # Promediar sobre la secuencia
        x = self.dropout(x)  # Aplicar dropout
        x = self.fc(x)
        return x

# 4. Preparación de datos con longitud variable
vocab = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '<pad>': 4}  # Añadimos el padding en un índice diferente (4)
padding_idx = vocab['<pad>']
min_seq_length = 50
max_seq_length = 100
num_samples = 10000
num_classes = 5

dataset = SyntheticSequenceDataset(num_samples, min_seq_length, max_seq_length)
train_data, test_data, train_labels, test_labels = train_test_split(dataset.data, dataset.labels, test_size=0.2, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=42)

train_dataset = SyntheticSequenceDataset(len(train_data), min_seq_length, max_seq_length)
train_dataset.data, train_dataset.labels = train_data, train_labels

val_dataset = SyntheticSequenceDataset(len(val_data), min_seq_length, max_seq_length)
val_dataset.data, val_dataset.labels = val_data, val_labels

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=SequenceCollator(vocab, padding_idx))
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=SequenceCollator(vocab, padding_idx))

# 5. Entrenamiento y validación
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SequenceClassifier(vocab_size=len(vocab), embed_dim=256, num_heads=8, num_classes=num_classes, num_layers=4, max_seq_length=max_seq_length).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler para reducir la tasa de aprendizaje
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Crear un input example como numpy.ndarray
input_example = torch.randint(0, len(vocab), (1, max_seq_length)).numpy()  # Convertir a numpy

# Inferir la firma (signature) basada en el input example y la salida del modelo
signature = mfs.infer_signature(input_example, model(torch.tensor(input_example).to(device)).detach().cpu().numpy())

# Registrar el modelo con el input example y la firma
mlflow.pytorch.log_model(model, "model", signature=signature, input_example=input_example)

# 6. Funciones de entrenamiento y validación
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for sequences, labels, max_len in loader:
        sequences, labels = sequences.to(device), labels.to(device)

        # Crear el padding mask para las secuencias
        padding_mask = (sequences == padding_idx)

        optimizer.zero_grad()
        outputs = model(sequences, padding_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return running_loss / len(loader), acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for sequences, labels, max_len in loader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Crear el padding mask para las secuencias
            padding_mask = (sequences == padding_idx)

            outputs = model(sequences, padding_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return running_loss / len(loader), acc

# 7. Registro con MLFlow
timestamp = datetime.now().strftime("%d-%m-%Y__%H-%M-%S")

# Finaliza cualquier ejecución activa antes de iniciar una nueva
if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_name=timestamp):
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("epochs", 100)

    for epoch in range(100):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Ajuste de la tasa de aprendizaje
        scheduler.step()

    mlflow.pytorch.log_model(model, "model")
