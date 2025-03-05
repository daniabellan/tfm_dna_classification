import argparse
import os
import yaml
from pathlib import Path
import numpy as np
import pywt
import time
from scipy.signal import butter, filtfilt, medfilt
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

from load_h5 import load_dict_h5


default_config = {
    "window_size": 1500,
    "step_ratio": 0.5,
    "sampling_rate": 4000,
    "kmers_size": 3,
    "num_samples": 1000
}

@dataclass
class SignalData:
    signal_pa: np.ndarray
    sequence: str
    label: int
    full_processed_signal: np.ndarray = None
    window_signal: np.ndarray = None
    kmers: np.ndarray = None


class SignalPreprocess:
    def __init__(self, 
                 sampling_rate:int,
                 window_size:int,
                 step_ratio:float):
        
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.step_ratio = step_ratio

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

    def wavelet_transform(self, signal, wavelet='db4', level=5, threshold=0.1):
        # Realizar la transformación wavelet
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Umbralización (eliminar coeficientes pequeños)
        coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        
        # Reconstruir la señal usando los coeficientes umbralizados
        filtered_signal = pywt.waverec(coeffs, wavelet)
        
        return filtered_signal    

    def modified_zscore(self, signal):
        median_X = np.median(signal)  # Mediana de la señal
        MAD = np.median(np.abs(signal - median_X))  # Desviación absoluta mediana (MAD)

        if MAD == 0:
            MAD = 1e-9  # Para evitar división por cero

        z_scores = (signal - median_X) / MAD  # Cálculo del Z-score modificado

        # Manejo de valores atípicos (outliers)
        threshold = 3.5  # Umbral de detección de outliers
        outliers = np.abs(z_scores) > threshold

        # Sustituir outliers con la media de los puntos adyacentes
        for i in np.where(outliers)[0]:
            if 1 <= i < len(signal) - 1:
                signal[i] = np.mean([signal[i - 1], signal[i + 1]])

        return z_scores

    # Preprocesar señal cargada/generada
    def preprocess_signal(self, signal, cutoff=1000, window_size=5, median_kernel_size=5):
        # Paso 1: Filtro de paso bajo (eliminar ruido de alta frecuencia)
        preprocessed_signal = self.lowpass_filter(signal, cutoff)
        
        # Paso 2: Filtro de mediana (eliminar picos de ruido impulsivo)
        preprocessed_signal = self.median_filter(preprocessed_signal, median_kernel_size)
        
        # Paso 3: Suavizado por media móvil (suavizar la señal)
        preprocessed_signal = self.moving_average(preprocessed_signal, window_size)
        
        # Paso 4: Transformada Wavelet para reducción de ruido
        preprocessed_signal = self.wavelet_transform(preprocessed_signal)

        # Paso 6: Aplicar Z-score para normalizar la señal
        preprocessed_signal = self.modified_zscore(preprocessed_signal)

        return np.array(preprocessed_signal, dtype=np.float32)

    def apply_sliding_window(self, signal):
        if self.window_size > len(signal) or self.window_size <= 0:
            raise ValueError("El tamaño de la ventana debe ser mayor que 0 y menor o igual a la longitud de la señal.")
        
        if not (0 < self.step_ratio <= 1):
            raise ValueError("El step_ratio debe estar entre 0 y 1.")
        
        step = max(1, int(self.window_size * self.step_ratio))
        num_windows = (len(signal) - self.window_size) // step + 1
        
        return np.array([signal[i : i + self.window_size] for i in range(0, len(signal) - self.window_size + 1, step)])


class RealDataset:
    def __init__(self,
                 datasets_path:str,
                 seed:int=42,
                 preprocess_data:bool=True):
        
        self.rng = np.random.default_rng(seed)

        self.window_size = default_config["window_size"]

        self.step_ratio = default_config["step_ratio"]

        self.sampling_rate = default_config["sampling_rate"]

        self.kmers_size = default_config["kmers_size"]

        self.prepr_fn = SignalPreprocess(sampling_rate = self.sampling_rate,
                                         window_size = self.window_size,
                                         step_ratio = self.step_ratio)

        self.kmer_dict = self.generate_kmer_dict()
        
        self.label_names = {}
        real_data_raw = []
        for class_idx, (data_name, dataset_path) in enumerate(datasets_path.items()): 
            real_data_raw.extend(self._load_fast5(fast5_path = dataset_path,
                                                  class_idx = class_idx))
            self.label_names[class_idx] = data_name

        self.real_data_sample = list(self.rng.choice(real_data_raw, 
                                                size = default_config["num_samples"], 
                                                replace = False))
        if preprocess_data:
            self._preprocess_fast5()

        pass

    def _preprocess_fast5(self):
        start = time.time()
        for read_idx, read_data in enumerate(self.real_data_sample):
            full_processed_signal = self.prepr_fn.preprocess_signal(signal = read_data.signal_pa)
            window_signal = self.prepr_fn.apply_sliding_window(signal = full_processed_signal)
            
            # Rellenar con datos el diccionario
            read_data.full_processed_signal = full_processed_signal
            read_data.window_signal = window_signal

            # Añadir K-Mers a la muestra del dataset real
            read_data.kmers = self.sequence_to_kmer_indices(sequence = read_data.sequence)

            if read_idx % 100 == 0 or read_idx==len(self.real_data_sample):
                print(f"Preprocessed read {read_idx}/{len(self.real_data_sample)}")

        print(f"Preprocessing done in {(time.time() - start):.4f} secs")


    def _load_fast5(self, 
                    fast5_path:str, 
                    class_idx:int,
                    h5_file:str = "final_data/matched_data.h5"):
        
        full_path = os.path.join(fast5_path, h5_file)
        reads = load_dict_h5(full_path)
        
        real_raw_data = []
        for read_idx, read_data in list(reads.items()):
            real_raw_data.append(
                SignalData(
                    signal_pa=read_data["signal_pa"],
                    sequence=read_data["sequence"],
                    label=class_idx
                )
            )
        
        return real_raw_data
    

    def sequence_to_kmer_indices(self, sequence):
        """
        Convierte una secuencia en una lista de índices de K-mers.
        - `sequence`: Secuencia de ADN (ej. "ACGTAG")
        - `k`: Tamaño del K-mer
        - `kmer_dict`: Diccionario {K-mer: índice}
        - `unk_idx`: Índice para K-mers desconocidos (opcional)
        """
        kmer_indices = []
        for i in range(len(sequence) - default_config["kmers_size"] + 1):
            kmer = sequence[i:i + default_config["kmers_size"]]
            if kmer in self.kmer_dict:
                kmer_indices.append(self.kmer_dict[kmer])
            else:
                raise "No K-Mer available. Hint: change K-size value"
        return kmer_indices

    def generate_kmer_dict(self):
        """Genera un diccionario de K-mers con sus índices únicos."""
        from itertools import product
        bases = ['A', 'C', 'G', 'T']
        kmer_list = [''.join(p) for p in product(bases, repeat=self.kmers_size)]
        kmer_dict = {kmer: idx for idx, kmer in enumerate(kmer_list)}
        return kmer_dict


class SimilarityChecker:
    def __init__(self,
                 loaded_dataset:RealDataset):
        
        self.rng = loaded_dataset.rng
        self.num_classes = len(set([x.label for x in loaded_dataset.real_data_sample]))
        self.label_names_dict =  loaded_dataset.label_names
        # self.labels = [x.label for x in loaded_dataset.real_data_sample]
        self.std_real_data_sample, self.labels = self.standardize_dataset(loaded_dataset.real_data_sample)

        pass

    def standardize_dataset(self, 
                            real_data_sample:list):
        min_label_count = np.inf
        for cls_num in range(self.num_classes):
            min_label = [x.label for x in real_data_sample].count(cls_num)
            if min_label < min_label_count:
                min_label_count = min_label

        balanced_samples = []
        balanced_labels = []
        for cls_num in range(self.num_classes):
            cls_samples = self.rng.choice([x for x in real_data_sample if x.label==cls_num], 
                                          size = min_label_count, 
                                          replace = False)
            balanced_samples.extend(cls_samples)
            balanced_labels.extend([cls_num] * len(cls_samples))
            
        # Find the max length of the signals
        max_length = max(len(sig) for sig in [x.signal_pa for x in balanced_samples])

        # Pad signal with zeros to normalize every signal to the same length
        signals_padded = np.array([np.pad(sig, (0, max_length - len(sig)), 'constant') for sig in [x.signal_pa for x in balanced_samples]])

        return signals_padded, balanced_labels

    def pca_similarity(self):
        scaler = StandardScaler()
        signals_scaled = scaler.fit_transform(self.std_real_data_sample)

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(signals_scaled)
        
        # Calcular la matriz de covarianza de los datos PCA
        cov_matrix = np.cov(X_pca.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        # Obtener centroides de cada clase
        centroids = {}
        for label_idx in self.label_names_dict.keys():
            centroids[label_idx] = X_pca[np.array(self.labels) == label_idx].mean(axis=0)

        # Calcular distancias entre centroides
        distances = {}
        label_keys = list(centroids.keys())

        for i in range(len(label_keys)):
            for j in range(i + 1, len(label_keys)):
                label_i, label_j = label_keys[i], label_keys[j]

                # Distancia Euclidiana
                euclidean_dist = np.linalg.norm(centroids[label_i] - centroids[label_j])

                # Distancia de Mahalanobis
                mahal_dist = mahalanobis(centroids[label_i], centroids[label_j], inv_cov_matrix)

                distances[(label_i, label_j)] = {
                    "euclidean": euclidean_dist,
                    "mahalanobis": mahal_dist
                }

        print("Distancias entre clústeres PCA:")
        for pair, dists in distances.items():
            label_names = [self.label_names_dict[pair[0]], self.label_names_dict[pair[1]]]
            print(f"Cluster {label_names[0]} vs {label_names[1]} -> Euclidean: {dists['euclidean']:.4f}, Mahalanobis: {dists['mahalanobis']:.4f}")

        # Graficar PCA con los centroides
        plt.figure(figsize=(8, 6))
        colors = ["red", "blue", "green"]
        for label_idx, color in zip(self.label_names_dict.keys(), colors):
            plt.scatter(X_pca[np.array(self.labels) == label_idx, 0], 
                        X_pca[np.array(self.labels) == label_idx, 1], 
                        label=f"{self.label_names_dict[label_idx]}", 
                        alpha=0.6,
                        color=color)

        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        plt.title("PCA - Distribución de Clústeres")
        plt.show()

    
    def tsne_similarity(self):
        scaler = StandardScaler()
        signals_scaled = scaler.fit_transform(self.std_real_data_sample)

        # Aplicar t-SNE
        tsne = TSNE(n_components=3, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(signals_scaled)
        
        # Calcular la matriz de covarianza de los datos t-SNE
        cov_matrix = np.cov(X_tsne.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        # Obtener centroides de cada clase
        centroids = {}
        for label_idx in self.label_names_dict.keys():
            centroids[label_idx] = X_tsne[np.array(self.labels) == label_idx].mean(axis=0)

        # Calcular distancias entre centroides
        distances = {}
        label_keys = list(centroids.keys())

        for i in range(len(label_keys)):
            for j in range(i + 1, len(label_keys)):
                label_i, label_j = label_keys[i], label_keys[j]

                # Distancia Euclidiana
                euclidean_dist = np.linalg.norm(centroids[label_i] - centroids[label_j])

                # Distancia de Mahalanobis
                mahal_dist = mahalanobis(centroids[label_i], centroids[label_j], inv_cov_matrix)

                distances[(label_i, label_j)] = {
                    "euclidean": euclidean_dist,
                    "mahalanobis": mahal_dist
                }

        print("\nDistancias entre clústeres en t-SNE:")
        for pair, dists in distances.items():
            label_names = [self.label_names_dict[pair[0]], self.label_names_dict[pair[1]]]
            print(f"Cluster {label_names[0]} vs {label_names[1]} -> Euclidean: {dists['euclidean']:.4f}, Mahalanobis: {dists['mahalanobis']:.4f}")

        # Graficar t-SNE con los centroides
        plt.figure(figsize=(8, 6))
        for label_idx in self.label_names_dict.keys():
            plt.scatter(X_tsne[np.array(self.labels) == label_idx, 0], 
                        X_tsne[np.array(self.labels) == label_idx, 1], 
                        label=f"{self.label_names_dict[label_idx]}", 
                        alpha=0.6)

        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.title("t-SNE - Distribución de Clústeres")
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a hybrid sequence classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    config_file = args.config

    try:
        # Load all dataset paths to check similarity
        datasets_path = yaml.safe_load(open(config_file, "r"))

    except:
        raise FileNotFoundError 
    
    # Load dataset
    loaded_dataset = RealDataset(datasets_path = datasets_path)
    
    # Instantiate class of similaritychecker
    sim_checker = SimilarityChecker(loaded_dataset)

    # Do PCA
    sim_checker.pca_similarity()

    # Do T-SNE
    sim_checker.tsne_similarity()
    pass