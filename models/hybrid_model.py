import torch
import torch.nn as nn
import torch.nn.functional as F
from models.residual_block import ResidualBlock


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        """
        Positional Encoding usando nn.Embedding para una implementación más eficiente.
        
        Args:
        - max_len (int): Longitud máxima de la secuencia.
        - d_model (int): Dimensión de la representación (debe coincidir con la salida del CNN antes del Transformer).
        """
        super(PositionalEncoding, self).__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.register_buffer("positions", torch.arange(max_len).unsqueeze(0))  # Índices de posición

    def forward(self, x):
        """
        Args:
        - x (Tensor): Entrada de tamaño (batch, seq_len, d_model)

        Returns:
        - Tensor con Positional Encoding aplicado.
        """
        seq_len = x.size(1)  # Obtener la longitud de la secuencia
        pos_enc = self.position_embedding(self.positions[:, :seq_len])  # Obtener las posiciones correspondientes
        return x + pos_enc  # Sumar las posiciones embebidas a la entrada

# Modelo Híbrido con Transformer para secuencias y procesamiento de señales eléctricas
class HybridSequenceClassifier(nn.Module):
    def __init__(self, 
                 input_channels,
                 vocab_size, 
                 embed_dim, 
                 num_heads, 
                 num_classes, 
                 num_layers, 
                 use_signals=True, 
                 use_sequences=True):
        super(HybridSequenceClassifier, self).__init__()

        # Usar señales y/o secuencias
        self.use_signals = use_signals
        self.use_sequences = use_sequences


        # --- Configuración de la rama para señales eléctricas ---
        # Convolución inicial para extracción de características locales
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, stride=3, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)  # Reducción de dimensionalidad

        # Bloques residuales para aumentar la profundidad de la red
        self.layer1 = ResidualBlock(64, 128, stride=2, dilation=1)
        self.layer2 = ResidualBlock(128, 256, stride=2, dilation=2)
        self.layer3 = ResidualBlock(256, 512, stride=2, dilation=4)
        self.layer4 = ResidualBlock(512, 1024, stride=2, dilation=8)

        # Pooling global para obtener características fijas
        self.global_pool_signals = nn.AdaptiveMaxPool1d(8)

        # Transformer en la rama de señales
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,  # La dimensionalidad de la señal después del CNN
            nhead=8,       # Número de cabezas de atención
            dim_feedforward=2048,  
            dropout=0.5, 
            batch_first=True
        )
        self.signal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2) 
        self.attention_layer = nn.Linear(1024, 1)
        self.positional_encoding = PositionalEncoding(max_len=1500, d_model=1024)  # max_len basado en window_size

        # --- Configuración de la rama para secuencias con Transformer ---
        # Embedding para representar tokens en un espacio d-dimensional
        self.kmer_padding_idx = vocab_size-1
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.kmer_padding_idx)

        # Conv1D para captura de patrones locales en K-mers
        self.seq_conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.seq_conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)

        # Normalización antes del Transformer
        self.seq_norm = nn.LayerNorm(embed_dim)
        
        # Capas del Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dropout
        self.seq_dropout = nn.Dropout(0.1)

        # Pooling de atención para reducir la dimensionalidad
        self.seq_attention_pooling = nn.Linear(embed_dim, 1)

        # --- Clasificadores ---
        # Clasificador conjunto
        self.fc1 = nn.Linear(512 * 8 + embed_dim * 8, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # Clasificador para señales únicamente
        self.fc1_signals = nn.Linear(512 * 2, 2048)
        self.fc2_signals = nn.Linear(2048, 512)
        self.fc3_signals = nn.Linear(512, num_classes)

        # Clasificador para secuencias únicamente
        self.fc1_sequences = nn.Linear(embed_dim, 2048)
        self.fc2_sequences = nn.Linear(2048, 512)
        self.fc3_sequences = nn.Linear(512, num_classes)
        
        # Regularización con Dropout
        self.fc1_dropout = nn.Dropout(0.6)
        self.fc2_dropout = nn.Dropout(0.6)

        # Normalizar esclas después de la concatenación
        self.bn_combined = nn.BatchNorm1d(512 * 8 + embed_dim * 8)


    def forward(self, signals, sequences, padding_mask=None):
        if self.use_signals:
            # --- Procesamiento de la rama de señales ---
            x_signals = F.relu(self.conv1(signals))
            x_signals = self.pool1(x_signals)

            # Pasar por los bloques residuales
            x_signals = self.layer1(x_signals)
            x_signals = self.layer2(x_signals)
            x_signals = self.layer3(x_signals)
            x_signals = self.layer4(x_signals)

            # Pooling global y aplanamiento
            x_signals = self.global_pool_signals(x_signals)
            # x_signals = x_signals.view(x_signals.size(0), -1)

            # Transformar el tamaño a (batch, seq_len, feature_dim) para el Transformer
            x_signals = x_signals.permute(0, 2, 1) 
            
            # Positional encoding
            x_signals = self.positional_encoding(x_signals)  
            
            # Transformer Encoder
            x_signals = self.signal_transformer(x_signals)

            # Aplicamos mean pooling para obtener una representación global
            x_signals = torch.mean(x_signals, dim=1)

            # --- Normalización L2 ---
            # Asegura que las características tengan una magnitud uniforme
            x_signals = F.normalize(x_signals, p=2, dim=1)

        if self.use_sequences:
            # --- Procesamiento de la rama de secuencias ---
            # Convertir -1 en máscara de padding (True donde hay padding)
            padding_mask = sequences == self.kmer_padding_idx

            # Embedding de K-mers
            x_sequences = self.embedding(sequences)

            # Convoluciones para capturar patrones locales
            x_sequences = x_sequences.permute(0, 2, 1)  # Cambiar a (batch_size, embed_dim, seq_length)
            x_sequences = F.relu(self.seq_conv1(x_sequences))
            x_sequences = F.relu(self.seq_conv2(x_sequences))
            x_sequences = x_sequences.permute(0, 2, 1)  # Volver a (batch_size, seq_length, embed_dim)

            # Normalización
            x_sequences = self.seq_norm(x_sequences)

            # Transformer Encoder
            x_sequences = self.transformer(x_sequences, src_key_padding_mask=padding_mask)

            # Attention Pooling para extraer características clave
            attention_scores = F.softmax(self.seq_attention_pooling(x_sequences), dim=1)  # (batch_size, seq_length, 1)
            x_sequences = (x_sequences * attention_scores).sum(dim=1)  # Sumar ponderadamente sobre la secuencia

        # --- Combinación y clasificación ---
        
        # Pasar por las capas completamente conectadas con activaciones ReLU y Dropout
        if self.use_sequences and self.use_signals:
            # Concatenar características de ambas ramas
            x = torch.cat((x_signals, x_sequences), dim=1)
           
            x = self.bn_combined(x)

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


