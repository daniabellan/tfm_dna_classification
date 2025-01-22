import torch
import torch.nn as nn
import torch.nn.functional as F
from models.residual_block import ResidualBlock


# Modelo Híbrido con Transformer para secuencias y procesamiento de señales eléctricas
class HybridSequenceClassifier(nn.Module):
    def __init__(self, 
                 input_channels,
                 vocab_size, 
                 embed_dim, 
                 num_heads, 
                 num_classes, 
                 num_layers, 
                 max_seq_length, 
                 use_signals=True, 
                 use_sequences=True):
        super(HybridSequenceClassifier, self).__init__()

        # Usar señales y/o secuencias
        self.use_signals = use_signals
        self.use_sequences = use_sequences


        # --- Configuración de la rama para señales eléctricas ---
        # Convolución inicial para extracción de características locales
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=19, stride=3)
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

        # --- Clasificadores ---
        # Clasificador conjunto
        self.fc1 = nn.Linear(512 * 8 + embed_dim * 8, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # Clasificador para señales únicamente
        self.fc1_signals = nn.Linear(512 * 8, 2048)
        self.fc2_signals = nn.Linear(2048, 512)
        self.fc3_signals = nn.Linear(512, num_classes)

        # Clasificador para secuencias únicamente
        self.fc1_sequences = nn.Linear(embed_dim * 8, 2048)
        self.fc2_sequences = nn.Linear(2048, 512)
        self.fc3_sequences = nn.Linear(512, num_classes)
        
        # Regularización con Dropout
        self.fc1_dropout = nn.Dropout(0.5)
        self.fc2_dropout = nn.Dropout(0.5)


    def forward(self, signals, sequences, padding_mask=None):
        if self.use_signals:
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

            # --- Normalización L2 ---
            # Asegura que las características tengan una magnitud uniforme
            x_signals = F.normalize(x_signals, p=2, dim=1)

        if self.use_sequences:
            # --- Procesamiento de la rama de secuencias ---
            # Embedding de secuencias y adición de codificación posicional
            # x_sequences = self.embedding(sequences) + self.positional_encoding[:, :sequences.size(1), :]
            x_sequences = self.embedding(sequences) 

            # Transformer Encoder
            x_sequences = self.transformer(x_sequences, src_key_padding_mask=padding_mask)

            # Pooling global sobre la salida del Transformer
            x_sequences = x_sequences.permute(0, 2, 1)  # Cambiar a (batch_size, embed_dim, seq_length)
            x_sequences = self.global_pool_signals(x_sequences)
            x_sequences = x_sequences.reshape(x_sequences.size(0), -1)

            # --- Normalización L2 ---
            # Asegura que las características tengan una magnitud uniforme
            x_sequences = F.normalize(x_sequences, p=2, dim=1)


        # --- Combinación y clasificación ---
        
        # Pasar por las capas completamente conectadas con activaciones ReLU y Dropout
        if self.use_sequences and self.use_signals:
            # Concatenar características de ambas ramas
            x = torch.cat((x_signals, x_sequences), dim=1)
           
            x = F.relu(self.fc1(x))
            x = self.fc1_dropout(x)
            x = F.relu(self.fc2(x))
            x = self.fc2_dropout(x)
            x = self.fc3(x)  
        
        elif self.use_signals:
            # Usar solo la rama de las señales
            x = F.relu(self.fc1_signals(x_signals))
            x = self.fc1_dropout(x)
            x = F.relu(self.fc2_signals(x))
            x = self.fc2_dropout(x)
            x = self.fc3_signals(x)  
       
        elif self.use_sequences:
            # Usar solo la rama de las secuencias
            x = F.relu(self.fc1_sequences(x_sequences))
            x = self.fc1_dropout(x)
            x = F.relu(self.fc2_sequences(x))
            x = self.fc2_dropout(x)
            x = self.fc3_sequences(x)  

        return x


