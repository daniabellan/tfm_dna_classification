import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import mlflow.models.signature as mfs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

# Clase ResidualBlock (sin cambios)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# Modelo Híbrido con Transformer para secuencias
class HybridSequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes, num_layers, max_seq_length):
        super(HybridSequenceClassifier, self).__init__()

        # Rama para señales eléctricas
        self.conv1 = nn.Conv1d(1, 32, kernel_size=19, stride=3)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)
        self.global_pool_signals = nn.AdaptiveMaxPool1d(8)

        # Clasificador conjunto
        self.fc1 = nn.Linear(512 * 8 + embed_dim * 8, 2048)  # Combina ambas ramas
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)  # Para 5 clases
        self.fc1_dropout = nn.Dropout(0.5)
        self.fc2_dropout = nn.Dropout(0.5)

        # Rama para secuencias con Transformer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_seq = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(0.5)  # Añadido para regularización


    def forward(self, signals, sequences, padding_mask=None):
        # Procesamiento de señales
        signals = signals.unsqueeze(1)  # Convertir a (batch_size, 1, signal_length) para Conv1d
        x_signals = signals.view(signals.size(0), 1, -1)  # Reajustar las dimensiones
        x_signals = F.relu(self.conv1(x_signals))
        x_signals = self.pool1(x_signals)
        x_signals = self.layer1(x_signals)
        x_signals = self.layer2(x_signals)
        x_signals = self.layer3(x_signals)
        x_signals = self.layer4(x_signals)
        x_signals = self.global_pool_signals(x_signals)
        x_signals = x_signals.view(x_signals.size(0), -1)

        # Procesamiento de k-mers con Transformer
        x_sequences = self.embedding(sequences) + self.positional_encoding[:, :sequences.size(1), :]  # Añadir codificación posicional
        x_sequences = self.transformer(x_sequences, src_key_padding_mask=padding_mask)
        x_sequences = x_sequences.permute(0, 2, 1)  # Para usar MaxPool1d
        x_sequences = self.global_pool_signals(x_sequences)
        x_sequences = x_sequences.reshape(x_sequences.size(0), -1)
        ##x_sequences = x_sequences.mean(dim=1)
        # sequences = self.dropout(sequences)
        ##x_sequences = self.fc_seq(x_sequences)

        # Solo rama secuencias
        # x_sequences = self.embedding(sequences) + self.positional_encoding[:, :sequences.size(1), :]
        # x_sequences = self.transformer(x_sequences, src_key_padding_mask=padding_mask)
        # x_sequences = x_sequences.mean(dim=1)  # Promediar sobre la secuencia
        # x_sequences = self.dropout(x_sequences)  # Aplicar dropout
        # x_sequences = self.fc_seq(x_sequences)

        # Normalización L2
        x_signals = F.normalize(x_signals, p=2, dim=1)  
        x_sequences = F.normalize(x_sequences, p=2, dim=1)
        
        # Combinar ambas ramas
        x = torch.cat((x_signals, x_sequences), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)  
        x = F.relu(self.fc2(x))
        x = self.fc2_dropout(x)  
        x = self.fc3(x)
        
        # Solo rama secuencias
        # x = sequences
        return x
    

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, window_size=1000, min_seq_length=50, max_seq_length=100):
        """
        Dataset Sintético adaptado para k-mers con ventana deslizante y padding.

        Args:
            num_samples: Número de muestras en el dataset.
            seq_length: Longitud de la secuencia base.
            num_kmers: Número total de k-mers diferentes (vocabulario).
            min_k: Longitud mínima de los k-mers.
            max_k: Longitud máxima de los k-mers.
            window_size: Tamaño de la ventana deslizante para segmentar la señal.
            padding_value: Valor usado para el padding en la señal.
        """
        self.num_samples = num_samples
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.window_size = window_size

        # Generar datos sintéticos
        self.signals, self.sequences, self.labels = self.generate_data()


    def generate_data(self):
        signals = []  # Señales eléctricas
        sequences, labels = self._generate_sequences()

        for seq_idx, sequence in enumerate(sequences):
            # print(f"Generated sequence [{seq_idx+1}/{len(sequences)}]")
            # Generar la señal asociada a la secuencia
            signal = self.generate_signal(sequence)

            # Aplicar ventana deslizante para segmentar la señal 
            segmented_signal = self.apply_sliding_window(signal)

            signals.append(segmented_signal)

        return signals, sequences, labels


    def _generate_sequences(self):
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

    def generate_signal(self, sequence):
        base_signals = {
            'A': lambda: np.random.uniform(low=10, high=11, size=50),
            'C': lambda: np.random.uniform(low=20, high=21, size=50),
            'T': lambda: np.random.uniform(low=30, high=31, size=50),
            'G': lambda: np.random.uniform(low=40, high=41, size=50)
        }
        signal = np.concatenate([base_signals[base]() for base in sequence])

        return signal

    def apply_sliding_window(self, signal):
        segmented_signal = []

        step_size = self.window_size

        for start in range(0, len(signal), step_size):
            end = start + step_size

            signal_segment = signal[start:end]
            if len(signal_segment) < step_size:
                signal_segment = np.pad(signal_segment, (0, step_size - len(signal_segment)), constant_values=0)

            segmented_signal.append(np.array(signal_segment, dtype=np.float32))


        return segmented_signal

    def add_noise(self, signal):
        noise = np.random.normal(loc=0.0, scale=self.noise_factor, size=signal.shape)
        signal_con_ruido = signal + noise
        return signal_con_ruido

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        sequences = self.sequences[idx]
        label = self.labels[idx]

        return signal, sequences, label


class SequenceCollatorSignals:
    def __init__(self, vocab, padding_idx=4):
        self.vocab = vocab
        self.padding_idx = padding_idx

    def __call__(self, batch):
        signals, sequences, labels = zip(*batch)

        # Encontrar la longitud máxima de secuencia en el batch
        max_len = max(len(seq) for seq in sequences)
        
        # Codificar las secuencias y hacer padding
        sequences_idx = []
        for seq in sequences:
            seq_idx = [self.vocab[base] for base in seq]
            # Aplicar padding a las secuencias para igualar las longitudes
            padded_seq = seq_idx + [self.padding_idx] * (max_len - len(seq))
            sequences_idx.append(padded_seq)

        # Padding a las señales
        padded_signals = []
        max_segments = max([len(sample[0]) for sample in batch])
        for signal in signals:
            # Rellenar señales con ceros para que todas tengan el mismo número de segmentos
            while len(signal) < max_segments:
                signal.append(np.zeros(len(signal[0]), dtype=np.float32))
            
            padded_signals.append(np.array(signal, dtype=np.float32))


        signal_tensor = torch.tensor(np.array(padded_signals), dtype=torch.float32)
        sequences_tensor = torch.tensor(sequences_idx, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return signal_tensor, sequences_tensor, labels_tensor

        # return torch.tensor(sequences_idx, dtype=torch.long), torch.tensor(labels, dtype=torch.long), max_len

# Callback de early stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Callback para detener el entrenamiento si la pérdida de validación no mejora después de un número de épocas.
        
        Args:
            patience (int): Número de épocas que espera antes de detenerse si no hay mejora.
            min_delta (float): Mínima mejora requerida para considerar un cambio como una mejora.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reiniciar contador si hay mejora
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Función para guardar y registrar la matriz de confusión
def log_confusion_matrix(model, loader, device, step):
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for signals, sequences, labels in loader:
            signals, sequences, labels = signals.to(device), sequences.to(device), labels.to(device)

            # Crear el padding mask para las secuencias
            padding_mask = (sequences == padding_idx)

            outputs = model(signals, sequences, padding_mask)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcular la matriz de confusión
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Guardar la figura como un archivo temporal
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Registrar en MLflow
    mlflow.log_artifact("confusion_matrix.png", f"confusion_matrix_step_{step}")


vocab = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '<pad>': 4}  # Añadimos el padding en un índice diferente (4)
padding_idx = vocab['<pad>']
min_seq_length = 100
max_seq_length = 150
num_samples = 1000
num_classes = 5

batch_size = 1024
learning_rate = 0.0001
epochs = 100

# Crear el dataset sintético
dataset = SyntheticDataset(num_samples=num_samples, min_seq_length=min_seq_length, max_seq_length=max_seq_length)

# Dividir el dataset en entrenamiento y validación usando train_test_split
train_signals, val_signals, train_sequences, val_sequences, train_labels, val_labels = train_test_split(
    dataset.signals, dataset.sequences, dataset.labels, test_size=0.2, random_state=42)

# Crear datasets de PyTorch para entrenamiento y validación
train_dataset = SyntheticDataset(num_samples=len(train_signals), min_seq_length=min_seq_length, max_seq_length=max_seq_length)
train_dataset.signals, train_dataset.sequences, train_dataset.labels = train_signals, train_sequences, train_labels

val_dataset = SyntheticDataset(num_samples=len(val_signals), min_seq_length=min_seq_length, max_seq_length=max_seq_length)
val_dataset.signals, val_dataset.sequences, val_dataset.labels = val_signals, val_sequences, val_labels

# Crear los DataLoader para entrenamiento y validación
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=SequenceCollatorSignals(vocab, padding_idx))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=SequenceCollatorSignals(vocab, padding_idx))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridSequenceClassifier(vocab_size=len(vocab), embed_dim=256, num_heads=4, num_classes=num_classes, num_layers=2, max_seq_length=max_seq_length).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Scheduler para reducir la tasa de aprendizaje
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Instanciar early stopping
early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

# Gradient clipping
max_grad_norm = 1.0  # Configurar un valor límite

# 6. Funciones de entrenamiento y validación
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch_idx, (signals, sequences, labels) in enumerate(loader):
        signals, sequences, labels = signals.to(device), sequences.to(device), labels.to(device)

        # Crear el padding mask para las secuencias
        padding_mask = (sequences == padding_idx)

        optimizer.zero_grad()
        outputs = model(signals, sequences, padding_mask)

        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping para evitar gradientes demasiado grandes e inestables
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    return running_loss / len(loader), acc, precision, recall, f1


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_idx, (signals, sequences, labels) in enumerate(loader):
            signals, sequences, labels = signals.to(device), sequences.to(device), labels.to(device)

            # Crear el padding mask para las secuencias
            padding_mask = (sequences == padding_idx)

            outputs = model(signals, sequences, padding_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return running_loss / len(loader), acc, precision, recall, f1


# 7. Registro con MLFlow
timestamp = datetime.now().strftime("%d-%m-%Y__%H-%M-%S")

# Finaliza cualquier ejecución activa antes de iniciar una nueva
if mlflow.active_run():
    mlflow.end_run()

# Configuración de la URI de MLflow (local o servidor remoto)
mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run(run_name=timestamp):
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("optimizer", optimizer.__class__.__name__)
    mlflow.log_param("scheduler_step_size", scheduler.step_size)
    mlflow.log_param("vocab_size", len(vocab))
    mlflow.log_param("seed", 42)

    for epoch in range(epochs):
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, device)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("train_precision", train_precision, step=epoch)
        mlflow.log_metric("train_recall", train_recall, step=epoch)
        mlflow.log_metric("train_f1", train_f1, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        mlflow.log_metric("val_precision", val_precision, step=epoch)
        mlflow.log_metric("val_recall", val_recall, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Precision={train_precision:.4f}, Recall={train_recall:.4f}, F1={train_f1:.4f}")
        print(f"              Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Precision={val_precision:.4f}, Recall={val_recall:.4f}, F1={val_f1:.4f}")

        # Ajuste de la tasa de aprendizaje
        scheduler.step()

        # Verificar el early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # Registrar la matriz de confusión cada 10 épocas
        if (epoch + 1) % 10 == 0 or early_stopping.early_stop:
            log_confusion_matrix(model, val_loader, device, epoch + 1)

    # Registrar el modelo final en MLflow
    mlflow.pytorch.log_model(model, "model")
    # Registrar la matriz de confusión final
    log_confusion_matrix(model, val_loader, device, "final")