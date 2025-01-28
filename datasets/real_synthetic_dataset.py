import yaml
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import zscore 
from datasets.load_h5 import load_dict_h5
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Plantilla para almacenar los datos
SIGNALDATA_KEYS = {
    "signal_pa": None,          # Señal cruda
    "processed_signal": None,   # Señal procesada
    "window_signal": None,      # Señal con ventana deslizante
    "sequence": None,           # Secuencia de nucleótidos
    "label": None               # Etiqueta de clase
}


class SignalGenerator:
    def __init__(self, 
                 rng, 
                 bases_per_second:int,
                 sampling_rate:int,
                 noise_factor:float = 0.1):
        
        self.rng = rng
        self.noise_factor = noise_factor
        self.bases_per_second = bases_per_second
        self.sampling_rate = sampling_rate

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


    def generate_signal(self, sequence):
        nucleotides = {
            'A': {'amplitude': 1.51, 'duration': 1 / self.bases_per_second},
            'C': {'amplitude': 1.52, 'duration': 1 / self.bases_per_second},
            'G': {'amplitude': 1.53, 'duration': 1 / self.bases_per_second},
            'T': {'amplitude': 1.54, 'duration': 1 / self.bases_per_second}
        }

        signal = []
        time = []
        current_time = 0

        for nucleotide in sequence:
            amplitude = nucleotides[nucleotide]['amplitude']
            duration = nucleotides[nucleotide]['duration']

            # Número de muestras para esta duración
            num_samples = int(duration * self.sampling_rate)

            # Generar segmento de señal
            segment = [amplitude] * num_samples
            segment_time = np.linspace(current_time, current_time + duration, num_samples, endpoint=False)

            # Agregar segmento a la señal completa
            signal.extend(segment)
            time.extend(segment_time)

            # Actualizar el tiempo actual
            current_time += duration

        # Añadir ruido gaussiano
        signal = self.add_noise(np.array(signal))

        # Añadir ruido salt and pepper
        signal = self.add_salt_and_pepper_noise(signal, salt_prob=0.02, pepper_prob=0.02)

        # Aplicar escalado aleatorio de amplitud
        signal = self.apply_random_amplitude_scaling(signal, num_segments=5, scale_range=(0.5, 1.5))

        return signal

class SignalPreprocess:
    def __init__(self, 
                 sampling_rate:int,
                 window_size:int):
        
        self.sampling_rate = sampling_rate
        self.window_size = window_size

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
    
    # Preprocesar señal cargada/generada
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


class RealSyntheticDataset:
    def __init__(self, 
                 config:dict, 
                 seed:int=42, 
                 mode:str="default", 
                 seq_variation_perc:float=0.3,
                 preprocess:bool=True):

        # Crear un generador de números aleatorios con una semilla fija
        self.rng = np.random.default_rng(seed)

        # Tamaño de la ventana deslizante
        self.window_size = config['window_size']
        
        # Distribución de las bases por "especie"
        self.base_probs = config["synthetic_dataset"]['base_probs']
        
        # Bases genéticas
        self.bases = config['bases']

        # Vocabulario (letras de las bases)
        self.vocab = config['vocab']

        # Número que representa el caracter del padding
        self.padding_idx = config['padding_idx']
        
        # Número de bases que pasan por el nanoporo por segundo.
        self.bases_per_second = config["bases_per_second"] 
        
        # Número de puntos de muestreo por segundo.
        self.sampling_rate = config["sampling_rate"]   

        # Almacenamos el modo de acceso al dataset
        self.mode = mode  

        # Indica si la señal del Dataset se va a preprocesar o no
        self.preprocess = preprocess 
        
        # Cargar funciones de pre-procesamiento de señal
        self.prepr_fn = SignalPreprocess(sampling_rate = self.sampling_rate,
                                         window_size = self.window_size)

        # Cargar funciones para generar señales sintéticas
        self.signal_gen_fn = SignalGenerator(rng = self.rng,
                                             bases_per_second = self.bases_per_second,
                                             sampling_rate = self.sampling_rate)


        # Número de clases dinámico en función de los datos reales y los sintéticos
        self.num_classes = len(config["real_dataset"]) + len(config["synthetic_dataset"]["base_probs"])

        # Número de muestras reales
        self.num_samples = config["num_samples"]

        # Cargar datos reales
        real_data = self._load_fast5(fast5_path = config["real_dataset"])

        # Crear una muestra de estos datos reales para evitar utilizar demasiados datos
        real_data_sample = list(self.rng.choice(real_data, 
                                                     size = self.num_samples, 
                                                     replace = False))
        
        # Configuracion para los datasets sintéticos utilizando los parámetros de la muestra real
        # Media de la longitud de la secuencia
        average_seq_length = np.average([len(x["sequence"]) for x in real_data_sample])
        
        # Longitud mínima y máxima de la secuencia +- X%
        self.min_seq_length = average_seq_length - int(average_seq_length * seq_variation_perc)
        self.max_seq_length = average_seq_length + int(average_seq_length * seq_variation_perc)
        
        # Generar datos sintéticos 
        synthetic_data = self._generate_data(real_data_sample)

        # Unir datasets (real + sintético)
        self.full_dataset = real_data_sample + synthetic_data

        # self.original_signal = [x["signal_pa"] for x in full_dataset]
        # self.processed_signal = [x["processed_signal"] for x in full_dataset]
        # self.window_signal = [x["window_signal"] for x in full_dataset]
        # self.sequences = [x["sequence"] for x in full_dataset]
        # self.labels = [x["label"] for x in full_dataset]

        pass

    
    def _load_fast5(self, fast5_path:str):
        real_data = []
        for class_idx, file in enumerate(fast5_path):
            reads = load_dict_h5(file)
            for read_idx, read_data in list(reads.items())[:20]:
                if self.preprocess:
                    processed_signal = self.prepr_fn.preprocess_signal(signal = read_data["signal_pa"])
                    window_signal = self.prepr_fn.apply_sliding_window(signal = processed_signal)
                else:
                    processed_signal = []
                    window_signal = self.prepr_fn.apply_sliding_window(signal = read_data["signal_pa"])


                # Rellenar con datos el diccionario
                signal_data = SIGNALDATA_KEYS.copy()  # Copiar la plantilla
                signal_data["signal_pa"] = read_data["signal_pa"]
                signal_data["processed_signal"] = processed_signal
                signal_data["window_signal"] = window_signal
                signal_data["sequence"] = read_data["sequence"]
                signal_data["label"] = class_idx

                real_data.append(signal_data)

        
        return real_data


    def _generate_data(self, real_data_sample):
        synthetic_data = []
        # Comprobar cuales son las etiquetas asignadas a los datos reales
        last_label_class = max(list(Counter([x["label"] for x in real_data_sample]).keys()))

        # Generar secuencias en base a las probabilidades de bases
        for label_offset, probs in enumerate(self.base_probs):
            # Hacer que el label comience desde last_label_class + 1
            label = last_label_class + 1 + label_offset
            
            for _ in range(self.num_samples):
                # Generar secuencia
                seq_length = self.rng.integers(self.min_seq_length, self.max_seq_length + 1)
                sequence = ''.join(self.rng.choice(self.bases, seq_length, p=probs))
                
                # Generar señal
                signal = self.signal_gen_fn.generate_signal(sequence = sequence)

                # Pre-procesar señal y aplicar sliding window
                if self.preprocess:
                    processed_signal = self.prepr_fn.preprocess_signal(signal = signal)
                    window_signal = self.prepr_fn.apply_sliding_window(signal = processed_signal)
                else:
                    processed_signal = []
                    window_signal = self.prepr_fn.apply_sliding_window(signal = signal)

                # Rellenar con datos el diccionario
                signal_data = SIGNALDATA_KEYS.copy()  # Copiar la plantilla
                signal_data["signal_pa"] = signal
                signal_data["processed_signal"] = processed_signal
                signal_data["window_signal"] = window_signal
                signal_data["sequence"] = sequence
                signal_data["label"] = label

                synthetic_data.append(signal_data)

                pass
        
        return synthetic_data


    # def __len__(self):
    #     return len(self.sequences)

    # def __getitem__(self, idx):
    #     if self.mode == "default":
    #         window_signal = self.window_signal[idx]
    #         sequences = self.sequences[idx]
    #         label = self.labels[idx]

    #         return window_signal, sequences, label
    #     if self.mode == "debug":
    #         original_signal = self.original_signal[idx]
    #         preprocessed_signals = self.processed_signal[idx]
    #         window_signal = self.window_signal[idx]
    #         sequences = self.sequences[idx]
    #         label = self.labels[idx]

            
    #         return preprocessed_signals, original_signal, window_signal, sequences, label


class StratifiedDataset(Dataset):
    def __init__(self, full_dataset, mode, config, seed=42):
        # Guardar datos completos
        self.full_dataset = full_dataset
        self.original_signal = [x["signal_pa"] for x in full_dataset]
        self.processed_signal = [x["processed_signal"] for x in full_dataset]
        self.window_signal = [x["window_signal"] for x in full_dataset]
        self.sequences = [x["sequence"] for x in full_dataset]
        self.labels = [x["label"] for x in full_dataset]

        self.seed = seed
        self.mode = mode 

        # Parámetros de la división
        # Crear datasets de PyTorch para entrenamiento y val
        self.train_ratio = config["train_split"]
        self.val_ratio = config["val_split"]
        self.test_ratio = config["test_split"]
        
        # Generar la división estratificada
        self.train_data, self.val_data, self.test_data = self._stratified_split(self.full_dataset, self.labels)

    def _stratified_split(self, dataset, labels):
        """
        Realiza una división estratificada en conjuntos de entrenamiento, validación y prueba.
        """
        # División inicial en train+val y test
        train_val_ratio = self.train_ratio + self.val_ratio
        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            dataset, labels, 
            test_size=self.test_ratio, 
            random_state=self.seed, 
            stratify=labels
        )

        # Proporción ajustada para dividir train y val
        val_adjusted_ratio = self.val_ratio / train_val_ratio
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_val_data, train_val_labels, 
            test_size=val_adjusted_ratio, 
            random_state=self.seed, 
            stratify=train_val_labels
        )

        return train_data, val_data, test_data



    def __len__(self):
        # El tamaño del dataset depende del modo actual
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.val_data)
        elif self.mode == 'test':
            return len(self.test_data)

    def __getitem__(self, idx):
        # Aquí seleccionamos qué conjunto devolver dependiendo del índice.
        if self.mode == 'train':
            dataset = self.train_data
        elif self.mode == 'val':
            dataset = self.val_data
        else:
            dataset = self.test_data
        
        # Retornar el índice del dataset correspondiente
        return dataset[idx]["window_signal"], dataset[idx]["sequence"], dataset[idx]["label"]


