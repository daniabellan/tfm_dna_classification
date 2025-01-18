import yaml
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import zscore 

class SyntheticDataset(Dataset):
    def __init__(self, config_file='config.yaml', mode="default", preprocess:bool=True):
        """
        Dataset Sintético adaptado para k-mers con ventana deslizante y padding.
        Carga las configuraciones desde un archivo YAML.

        Args:
            config_file: Ruta al archivo YAML que contiene la configuración.
        """
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.num_samples = config['dataset']['num_samples']
        self.min_seq_length = config['dataset']['min_seq_length']
        self.max_seq_length = config['dataset']['max_seq_length']
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

        return preprocessed_signals, original_signals, window_signals, sequences, labels

    def _generate_sequences(self):
        data, labels = [], []

        for label, probs in enumerate(self.base_probs):
            for _ in range(self.num_samples // len(self.base_probs)):
                seq_length = np.random.randint(self.min_seq_length, self.max_seq_length + 1)
                sequence = ''.join(np.random.choice(self.bases, seq_length, p=probs))
                data.append(sequence)
                labels.append(label)

        return data, labels
    
    """
    # OLD GENERATION OF SIGNALS
    def generate_signal(self, sequence):
        base_signals = {
            'A': lambda: np.random.uniform(low=10, high=11, size=50),
            'C': lambda: np.random.uniform(low=20, high=21, size=50),
            'T': lambda: np.random.uniform(low=30, high=31, size=50),
            'G': lambda: np.random.uniform(low=40, high=41, size=50)
        }
        signal = np.concatenate([base_signals[base]() for base in sequence])

        return signal
    """
        
    # NEW GENERATION SIGNALS ~REAL
    def generate_signal(self, sequence):
        time_points = np.linspace(0, self.total_time, int(self.sampling_rate * self.total_time))

        nucleotides = {
            'A': {'amplitude': 1.5, 'duration': 0.4},  # Amplitud y duración aproximada de la fluctuación
            'C': {'amplitude': 1.2, 'duration': 0.35},
            'G': {'amplitude': 1.7, 'duration': 0.45},
            'T': {'amplitude': 1.3, 'duration': 0.4}
        }

        # Generar la señal sintética
        signal = np.zeros_like(time_points)

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

        # Añadir ruido a la señal
        noise = np.random.normal(0, 0.3, len(signal))  # Ruido gaussiano
        signal += noise

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

    def add_noise(self, signal):
        noise = np.random.normal(loc=0.0, scale=self.noise_factor, size=signal.shape)
        return signal + noise

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


