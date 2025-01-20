import torch
import torch.nn as nn
import torch.nn.functional as F
from models.residual_block import ResidualBlock


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query_fc = nn.Linear(embed_dim, embed_dim)  # Proyección para la rama de señales
        self.key_fc = nn.Linear(embed_dim, embed_dim)    # Proyección para la rama de secuencias
        self.value_fc = nn.Linear(embed_dim, embed_dim)  # Proyección para la rama de secuencias
        
    def forward(self, x_signals, x_sequences):
        """
        x_signals: Salida de la rama de señales
        x_sequences: Salida de la rama de secuencias
        """
        # Calcular las consultas (queries), claves (keys) y valores (values) para la atención cruzada
        Q = self.query_fc(x_signals)  # Proyección de las señales
        K = self.key_fc(x_sequences)  # Proyección de las secuencias
        V = self.value_fc(x_sequences)  # Proyección de las secuencias

        # Calcular la atención cruzada
        attention_scores = torch.matmul(Q, K.transpose(1, 2))  # Producto punto entre queries y claves
        attention_weights = F.softmax(attention_scores, dim=-1)  # Normalización con softmax

        # Aplicar la atención a los valores
        attended_features = torch.matmul(attention_weights, V)  # Producto punto con los valores

        return attended_features

# Modelo Híbrido con Transformer para secuencias y procesamiento de señales eléctricas
class HybridSequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes, num_layers, max_seq_length):
        super(HybridSequenceClassifier, self).__init__()

        # --- Configuración de la rama para señales eléctricas ---
        # Convolución inicial para extracción de características locales
        self.conv1 = nn.Conv1d(10, 32, kernel_size=19, stride=3)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)  # Reducción de dimensionalidad

        # Bloques residuales para aumentar la profundidad de la red
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)

        # Pooling global para obtener características fijas
        self.global_pool_signals = nn.AdaptiveMaxPool1d(8)

        # --- Configuración de la rama para secuencias con Transformer ---
        # Embedding para representar tokens en un espacio d-dimensional
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Codificación posicional para incorporar información de posición en la secuencia
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))

        # Capas del Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Clasificador conjunto ---
        # Combina características de ambas ramas para predecir las clases
        self.fc1 = nn.Linear(512 * 8 + embed_dim * 8, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # Regularización con Dropout
        self.fc1_dropout = nn.Dropout(0.5)
        self.fc2_dropout = nn.Dropout(0.5)

        # Atención cruzada entre ramas
        self.cross_attention = CrossAttention(embed_dim)

    def forward(self, signals, sequences, padding_mask=None):
        # --- Procesamiento de la rama de señales ---
        # signals = signals.unsqueeze(1)  # (batch_size, 1, signal_length) para Conv1d
        x_signals = F.relu(self.conv1(signals))
        x_signals = self.pool1(x_signals)

        # Pasar por los bloques residuales
        x_signals = self.layer1(x_signals)
        x_signals = self.layer2(x_signals)
        x_signals = self.layer3(x_signals)
        x_signals = self.layer4(x_signals)

        # Pooling global y aplanamiento
        x_signals = self.global_pool_signals(x_signals)
        x_signals = x_signals.view(x_signals.size(0), -1)

        # --- Procesamiento de la rama de secuencias ---
        # Embedding de secuencias y adición de codificación posicional
        x_sequences = self.embedding(sequences) + self.positional_encoding[:, :sequences.size(1), :]

        # Transformer Encoder
        x_sequences = self.transformer(x_sequences, src_key_padding_mask=padding_mask)

        # Pooling global sobre la salida del Transformer
        x_sequences = x_sequences.permute(0, 2, 1)  # Cambiar a (batch_size, embed_dim, seq_length)
        x_sequences = self.global_pool_signals(x_sequences)
        x_sequences = x_sequences.reshape(x_sequences.size(0), -1)

        # --- Normalización L2 ---
        # Asegura que las características tengan una magnitud uniforme
        x_signals = F.normalize(x_signals, p=2, dim=1)
        x_sequences = F.normalize(x_sequences, p=2, dim=1)

        # --- Atención cruzada entre las ramas --- 
        # x_signals_attention = self.cross_attention(x_signals, x_sequences)
        # x_sequences_attention = self.cross_attention(x_sequences, x_signals)

        # # Combinación de características con atención cruzada
        # x = torch.cat((x_signals_attention, x_sequences_attention), dim=1)

        # --- Combinación y clasificación ---
        # Concatenar características de ambas ramas
        x = torch.cat((x_signals, x_sequences), dim=1)

        # Pasar por las capas completamente conectadas con activaciones ReLU y Dropout
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_dropout(x)
        x = self.fc3(x)  # Predicción final

        return x


