import yaml
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import zscore 
from dataclasses import dataclass
from datasets.load_h5 import load_dict_h5

@dataclass
class SignalData:
    signal_pa: list
    sequence: list

class RealSyntheticDataset(Dataset):
    def __init__(self, config:dict, seed=42, mode="default", preprocess:bool=True):
        """
        Dataset Sintético adaptado para k-mers con ventana deslizante y padding.
        Carga las configuraciones desde un archivo YAML.

        Args:
            config_file: Ruta al archivo YAML que contiene la configuración.
        """
        # Crear un generador de números aleatorios con una semilla fija
        self.rng = np.random.default_rng(seed)

        real_data = []
        for file in config["real_dataset"]:
            reads = load_dict_h5(file)
            for read_idx, read_data in reads.items():
                pass
                real_data.append(
                    SignalData(signal_pa = read_data["signal_pa"],
                               sequence = read_data["sequence"])
                )
        
        self.real_data_sample = self.rng.choice(real_data, size=50, replace=False)
        max_signal_length = max([len(x.signal_pa) for x in self.real_data_sample])
        max_sequence_length = max([len(x.sequence) for x in self.real_data_sample])

        self.num_samples = len(self.real_data_sample)
        self.min_seq_length = max_sequence_length - int(max_sequence_length * 0.1)
        self.max_seq_length = max_sequence_length
        self.window_size = config['dataset']['window_size']
        self.noise_factor = config['dataset'].get('noise_factor', 0.0)
        self.base_probs = config['dataset']['base_probs']
        self.bases = config['dataset']['bases']
        
        self.vocab = config['vocab']
        self.padding_idx = config['padding_idx']

        self.mode = mode  # Almacenamos el modo de acceso

        self.sampling_rate = 1000  # Frecuencia de muestreo en Hz (10 kHz)
        self.total_time = 1.0  # Duración total de la señal en segundos

        self.preprocess = preprocess # Preprocess signal



        # Generar datos sintéticos
        self.preprocessed_signals, self.complete_signals, self.signals, self.sequences, self.labels = self.generate_data()

    def generate_data(self):
        preprocessed_signals = []
        original_signals = []
        window_signals = []  # Señales eléctricas divididas
        sequences, labels = self._generate_sequences()

        # Procesar señales sintéticas
        for seq_idx, sequence in enumerate(sequences):
            # Generar la señal asociada a la secuencia
            signal = self.generate_signal(sequence)
            original_signals.append(signal)

            # Preprocess signals
            if self.preprocess:
                preprocessed_signal = self.preprocess_signal(signal)
                preprocessed_signals.append(preprocessed_signal)

                # Aplicar ventana deslizante para segmentar la señal 
                segmented_signal = self.apply_sliding_window(preprocessed_signal)
            else:
                # Aplicar ventana deslizante para segmentar la señal 
                segmented_signal = self.apply_sliding_window(signal)

            window_signals.append(segmented_signal)

        # Procesar señales reales
        next_label_class = max(list(Counter(labels).keys()))
        for data in self.real_data_sample:
            # Añadir clase y secuencia asociada a la señal real
            labels.append(next_label_class)
            sequences.append(data.sequence)
            original_signals.append(data.signal_pa)

            # Preprocess signals
            if self.preprocess:
                preprocessed_signal = self.preprocess_signal(data.signal_pa)
                preprocessed_signals.append(preprocessed_signal)

                # Aplicar ventana deslizante para segmentar la señal 
                segmented_signal = self.apply_sliding_window(preprocessed_signal)
            else:
                # Aplicar ventana deslizante para segmentar la señal 
                segmented_signal = self.apply_sliding_window(data.signal_pa)

            window_signals.append(segmented_signal)

        return preprocessed_signals, original_signals, window_signals, sequences, labels

    def _generate_sequences(self):
        data, labels = [], []

        for label, probs in enumerate(self.base_probs):
            for _ in range(self.num_samples // len(self.base_probs)):
                seq_length = self.rng.integers(self.min_seq_length, self.max_seq_length + 1)
                sequence = ''.join(self.rng.choice(self.bases, seq_length, p=probs))
                data.append(sequence)
                labels.append(label)

        return data, labels

    def add_salt_and_pepper_noise(self, signal, salt_prob=0.02, pepper_prob=0.02):
        noisy_signal = signal.copy()
        num_points = len(signal)
        
        # Generar índices aleatorios para ruido 'salt' (valores altos)
        salt_indices = self.rng.choice(num_points, int(salt_prob * num_points), replace=False)
        noisy_signal[salt_indices] = np.max(signal) * 1.5  # Valor máximo (salt)
        
        # Generar índices aleatorios para ruido 'pepper' (valores bajos)
        pepper_indices = self.rng.choice(num_points, int(pepper_prob * num_points), replace=False)
        noisy_signal[pepper_indices] = np.min(signal) * 1.5  # Valor mínimo (pepper)
        
        return noisy_signal

    def apply_random_amplitude_scaling(self, signal, num_segments=5, scale_range=(0.5, 1.5)):
        scaled_signal = signal.copy()
        segment_length = len(signal) // num_segments
        
        for i in range(num_segments):
            # Definir inicio y fin del segmento
            start = i * segment_length
            end = start + segment_length if (i < num_segments - 1) else len(signal)
            
            # Escalado aleatorio
            scale_factor = self.rng.uniform(scale_range[0], scale_range[1])
            scaled_signal[start:end] *= scale_factor
        
        return scaled_signal

    def add_noise(self, signal):
        noise = np.random.normal(loc=0.0, scale=self.noise_factor, size=signal.shape)
        return signal + noise

    # NEW GENERATION SIGNALS ~REAL
    def generate_signal(self, sequence):
        time_points = np.linspace(0, self.total_time, int(self.sampling_rate * self.total_time))

        nucleotides = {
            'A': {'amplitude': 1.51, 'duration': 0.4},  # Amplitud y duración aproximada de la fluctuación
            'C': {'amplitude': 1.52, 'duration': 0.4},
            'G': {'amplitude': 1.53, 'duration': 0.4},
            'T': {'amplitude': 1.54, 'duration': 0.4}
        }

        # Generar la señal sintética
        signal = np.zeros_like(time_points, dtype=np.float32)

        # Inicializar el índice del tiempo
        current_time = 0
        for nucleotide in sequence:
            # Características del nucleótido
            amplitude = nucleotides[nucleotide]['amplitude']
            duration = nucleotides[nucleotide]['duration']
            
            # Generar pulso rectangular para este nucleótido
            end_time = current_time + duration
            pulse_time_points = time_points[(time_points >= current_time) & (time_points <= end_time)]
            
            # Generar el pulso rectangular
            pulse = amplitude * np.ones_like(pulse_time_points)  # Pulso constante durante la duración
            signal[(time_points >= current_time) & (time_points <= end_time)] = pulse
            
            # Avanzar al siguiente tiempo
            current_time = end_time

        # Añadir ruido gaussiano a la señal
        noise = self.rng.normal(0.9, 1, len(signal))
        signal += noise

        # Añadir ruido salt and pepper
        signal = self.add_salt_and_pepper_noise(signal, salt_prob=0.02, pepper_prob=0.02)

        # Aplicar escalado aleatorio de amplitud
        signal = self.apply_random_amplitude_scaling(signal, num_segments=5, scale_range=(0.5, 1.5))


        return signal

    # Función de filtro de paso bajo
    def lowpass_filter(self, data, cutoff, order=4):
        nyquist = 0.5 * self.sampling_rate  # Frecuencia de Nyquist
        # Asegurarse de que la frecuencia de corte no supere la frecuencia de Nyquist
        normal_cutoff = min(cutoff / nyquist, 0.99)  # Normalizar la frecuencia de corte y asegurarse de que no sea >= 1
        b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Crear el filtro
        return filtfilt(b, a, data)  # Aplicar el filtro

    # Filtro de mediana para eliminar ruido impulsivo
    def median_filter(self, data, kernel_size):
        return medfilt(data, kernel_size)

    # Filtro de media móvil
    def moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    def preprocess_signal(self, signal, cutoff=1000, window_size=5, median_kernel_size=5):
        # Paso 1: Filtro de paso bajo (eliminar ruido de alta frecuencia)
        preprocessed_signal = self.lowpass_filter(signal, cutoff)
        
        # # Paso 2: Filtro de mediana (eliminar picos de ruido impulsivo)
        preprocessed_signal = self.median_filter(preprocessed_signal, median_kernel_size)
        
        # # Paso 3: Suavizado por media móvil (suavizar la señal)
        preprocessed_signal = self.moving_average(preprocessed_signal, window_size)

        # Paso 4: Aplicar Z-score para normalizar la señal
        preprocessed_signal = zscore(preprocessed_signal)

        return preprocessed_signal

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



    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.mode == "default":
            signal = self.signals[idx]
            sequences = self.sequences[idx]
            label = self.labels[idx]

            return signal, sequences, label
        
        if self.mode == "debug":
            original_signal = self.complete_signals[idx]
            preprocessed_signals = self.preprocessed_signals[idx]
            signal = self.signals[idx]
            sequences = self.sequences[idx]
            label = self.labels[idx]

            
            return preprocessed_signals, original_signal, signal, sequences, label


