import torch
import torch.nn as nn
import torch.nn.functional as F
from models.residual_block import ResidualBlock  # Asegúrate de que la ruta sea la correcta


###########################################
#            UTILIDADES                   #
###########################################

class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        """
        Implementación del Positional Encoding usando nn.Embedding.

        Args:
            max_len (int): Longitud máxima de la secuencia.
            d_model (int): Dimensión de la representación (debe coincidir con la salida del CNN antes del Transformer).
        """
        super(PositionalEncoding, self).__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.register_buffer("positions", torch.arange(max_len).unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x (Tensor): Entrada de tamaño (batch, seq_len, d_model)

        Returns:
            Tensor con Positional Encoding aplicado.
        """
        seq_len = x.size(1)
        pos_enc = self.position_embedding(self.positions[:, :seq_len])
        return x + pos_enc


###########################################
#         RAMA DE SEÑALES (SIGNALS)        #
###########################################

class SignalBranch(nn.Module):
    def __init__(self, input_channels, max_len=1500, transformer_layers=2, transformer_dropout=0.5):
        """
        Rama para el procesamiento de señales eléctricas.

        Args:
            input_channels (int): Número de canales de entrada.
            max_len (int): Longitud máxima para el Positional Encoding.
            transformer_layers (int): Número de capas en el Transformer Encoder.
            transformer_dropout (float): Dropout para el Transformer Encoder.
        """
        super(SignalBranch, self).__init__()

        # Extracción de características locales con convolución y pooling
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, stride=3, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        # Bloques residuales
        self.layer1 = ResidualBlock(64, 128, stride=2, dilation=1)
        self.layer2 = ResidualBlock(128, 256, stride=2, dilation=2)
        self.layer3 = ResidualBlock(256, 512, stride=2, dilation=4)
        self.layer4 = ResidualBlock(512, 1024, stride=2, dilation=8)

        # Pooling global para obtener características fijas
        self.global_pool = nn.AdaptiveMaxPool1d(8)

        # Positional encoding y Transformer para capturar relaciones a nivel de secuencia
        self.positional_encoding = PositionalEncoding(max_len=max_len, d_model=1024)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=8,
            dim_feedforward=2048,
            dropout=transformer_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def forward(self, signals):
        x = F.relu(self.conv1(signals))
        x = self.pool1(x)
        
        # Pasar por los bloques residuales
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pooling global y preparación para el Transformer
        x = self.global_pool(x)         # (batch, channels, seq_len_fixed)
        x = x.permute(0, 2, 1)            # (batch, seq_len_fixed, channels)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        
        # Agregación a nivel de secuencia y normalización L2
        x = torch.mean(x, dim=1)         # (batch, 1024)
        x = F.normalize(x, p=2, dim=1)
        return x


###########################################
#       RAMA DE SECUENCIAS (SEQUENCES)      #
###########################################

class SequenceBranch(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, kmer_padding_idx=None):
        """
        Rama para el procesamiento de secuencias con Transformer.

        Args:
            vocab_size (int): Tamaño del vocabulario (número de tokens).
            embed_dim (int): Dimensión del embedding.
            num_heads (int): Número de cabezas de atención en el Transformer.
            num_layers (int): Número de capas del Transformer.
            kmer_padding_idx (int, opcional): Índice de padding para el embedding. 
                                              Por defecto se toma vocab_size - 1.
        """
        super(SequenceBranch, self).__init__()

        self.kmer_padding_idx = vocab_size - 1 if kmer_padding_idx is None else kmer_padding_idx

        # Embedding y convoluciones para capturar patrones locales
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.kmer_padding_idx)
        self.seq_conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.seq_conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.seq_norm = nn.LayerNorm(embed_dim)

        # Transformer Encoder para capturar dependencias a larga distancia
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Módulo de Attention Pooling para extraer características clave
        self.seq_attention_pooling = nn.Linear(embed_dim, 1)
        self.seq_dropout = nn.Dropout(0.1)

    def forward(self, sequences):
        # Crear máscara de padding
        padding_mask = sequences == self.kmer_padding_idx

        # Embedding
        x = self.embedding(sequences)  # (batch, seq_len, embed_dim)

        # Convoluciones (se necesita transponer para Conv1d)
        x = x.permute(0, 2, 1)         # (batch, embed_dim, seq_len)
        x = F.relu(self.seq_conv1(x))
        x = F.relu(self.seq_conv2(x))
        x = x.permute(0, 2, 1)         # (batch, seq_len, embed_dim)

        # Normalización y Transformer Encoder
        x = self.seq_norm(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Attention Pooling: extrae una representación global de la secuencia
        attention_scores = F.softmax(self.seq_attention_pooling(x), dim=1)  # (batch, seq_len, 1)
        x = (x * attention_scores).sum(dim=1)  # (batch, embed_dim)
        return x


class CNNKmerBranch(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes=[3, 5, 7]):
        super(CNNKmerBranch, self).__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 1)

        # Convoluciones con diferentes tamaños de ventana
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k//2) for k in kernel_sizes
        ])
        
        # Capa de Pooling Global
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Proyección final
        self.fc = nn.Linear(num_filters * len(kernel_sizes), embed_dim)

    def forward(self, sequences):
        x = self.embedding(sequences)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len) -> Formato necesario para Conv1D

        # Aplicar cada conv y ReLU
        conv_outputs = [F.relu(conv(x)) for conv in self.convs]

        # Pooling global y concatenación
        pooled_outputs = [self.global_pool(conv).squeeze(2) for conv in conv_outputs]
        x = torch.cat(pooled_outputs, dim=1)

        # Proyección final a embed_dim
        # x = self.fc(x)
        return x

###########################################
#         CLASIFICADORES (HEADS)          #
###########################################

class CombinedClassifier(nn.Module):
    def __init__(self, signals_dim, sequences_dim, num_classes):
        """
        Clasificador para cuando se usan ambas ramas.

        Args:
            signals_dim (int): Dimensión de la representación de la rama de señales.
            sequences_dim (int): Dimensión de la representación de la rama de secuencias.
            num_classes (int): Número de clases de salida.
        """
        super(CombinedClassifier, self).__init__()
        input_dim = signals_dim + sequences_dim
        self.bn = nn.BatchNorm1d(1024 + 768)
        self.fc1 = nn.Linear(1024 + 768, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class SignalsClassifier(nn.Module):
    def __init__(self, signals_dim, num_classes):
        """
        Clasificador para usar solo la rama de señales.

        Args:
            signals_dim (int): Dimensión de la representación de la rama de señales.
            num_classes (int): Número de clases de salida.
        """
        super(SignalsClassifier, self).__init__()
        self.fc1 = nn.Linear(signals_dim, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class SequencesClassifier(nn.Module):
    def __init__(self, num_filters, kernel_sizes, num_classes):
        """
        Clasificador para usar solo la rama de secuencias con CNN.

        Args:
            num_filters (int): Número de filtros por cada tamaño de kernel en la CNN.
            kernel_sizes (list): Lista con los tamaños de los kernels utilizados en la CNN.
            num_classes (int): Número de clases de salida.
        """
        super(SequencesClassifier, self).__init__()
        
        # El tamaño de entrada es num_filters * len(kernel_sizes), 
        # porque concatenamos todas las salidas de las convoluciones
        input_dim = num_filters * len(kernel_sizes)

        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


###########################################
#     MODELO HIBRIDO COMPLETO           #
###########################################

class HybridSequenceClassifier(nn.Module):
    def __init__(self, 
                 input_channels,
                 vocab_size, 
                 embed_dim, 
                 num_heads, 
                 num_classes, 
                 num_layers, 
                 use_signals=True, 
                 use_sequences=True,
                 max_len=1500):
        """
        Modelo híbrido que combina procesamiento de señales eléctricas y secuencias.

        Args:
            input_channels (int): Número de canales de entrada para señales.
            vocab_size (int): Tamaño del vocabulario para secuencias.
            embed_dim (int): Dimensión del embedding para secuencias.
            num_heads (int): Número de cabezas en el Transformer de secuencias.
            num_classes (int): Número de clases de salida.
            num_layers (int): Número de capas en el Transformer de secuencias.
            use_signals (bool): Si se activa la rama de señales.
            use_sequences (bool): Si se activa la rama de secuencias.
            max_len (int): Longitud máxima para el Positional Encoding en la rama de señales.
        """
        super(HybridSequenceClassifier, self).__init__()
        self.use_signals = use_signals
        self.use_sequences = use_sequences

        if use_signals:
            self.signal_branch = SignalBranch(input_channels, max_len=max_len)
        if use_sequences:
            # self.sequence_branch = SequenceBranch(vocab_size, embed_dim, num_heads, num_layers)
            self.num_filters=256
            self.kernel_sizes=[3, 5, 7]
            self.sequence_branch = CNNKmerBranch(vocab_size=vocab_size, 
                                                 embed_dim=embed_dim, 
                                                 num_filters=self.num_filters,
                                                 kernel_sizes=self.kernel_sizes)

        # Seleccionar el clasificador según las ramas utilizadas
        if use_signals and use_sequences:
            self.classifier = CombinedClassifier(signals_dim=1024, sequences_dim=embed_dim, num_classes=num_classes)
        elif use_signals:
            self.classifier = SignalsClassifier(signals_dim=1024, num_classes=num_classes)
        elif use_sequences:
            self.classifier = SequencesClassifier(num_filters=self.num_filters,
                                                  kernel_sizes=self.kernel_sizes,
                                                  num_classes=num_classes)

    def forward(self, signals, sequences, padding_mask=None):
        outputs = []
        if self.use_signals:
            x_signals = self.signal_branch(signals)
            outputs.append(x_signals)
        if self.use_sequences:
            # Asumiendo que modificas SequenceBranch para aceptar padding_mask
            # x_sequences = self.sequence_branch(sequences, padding_mask)
            x_sequences = self.sequence_branch(sequences)
            outputs.append(x_sequences)
        
        if len(outputs) > 1:
            x = torch.cat(outputs, dim=1)
        else:
            x = outputs[0]
        
        x = self.classifier(x)
        return x
