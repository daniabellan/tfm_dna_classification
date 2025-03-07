import torch
from torch.utils.data import ConcatDataset
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

from src.dataset.load_dataset import RealDataset
from src.dataset.collator import SequenceSignalsCollator
from src.utils.common import parse_args, load_config
from src.dataset.domain.utils import generate_kmer_dict

# Random generation seed
seed = 42
torch.manual_seed(seed)


class SimilarityChecker:
    def __init__(self, loaded_dataset: RealDataset):
        
        # Random number generator from the first dataset
        self.rng = loaded_dataset.datasets[0].rng

        # K-Mer size
        self.kmer_size = loaded_dataset.datasets[0].kmers_size

        # Extract unique class names from dataset paths
        # Also extracts labels per each signal
        self.labels = []
        self.sequences = []
        self.label_names_dict = {}
        for dataset in loaded_dataset.datasets:
            self.label_names_dict[dataset.class_idx] = dataset.dataset_path.split("/")[-3]
            for signal in dataset.data:
                self.labels.append(signal.label)
                self.sequences.append(signal.sequence)
        
        # Number of classes
        self.num_classes = len(self.label_names_dict)

        # Extract and standardize signals (only pre-processed signals)
        full_processed_signals = []
        for dataset in loaded_dataset:
            full_processed_signals.append(dataset["full_processed_signal"])

        self.std_signals = self.standardize_dataset(full_processed_signals)

        # Create kmer frequencies
        self.kmer_frequencies = np.array([self.compute_kmer_frequencies(seq) for seq in self.sequences])


    def compute_kmer_frequencies(self, sequence:str):
        # Generate kmer dictionary that represents kmers
        kmer_dict = generate_kmer_dict(self.kmer_size)
        
        vector = np.zeros(len(kmer_dict))  # Vector de frecuencia de K-mers
        for i in range(len(sequence) - self.kmer_size + 1):
            kmer = sequence[i:i + self.kmer_size]
            if kmer in kmer_dict:
                vector[kmer_dict[kmer]] += 1  # Contamos la aparición del K-mer
        return vector / np.sum(vector)  # Normalizamos por la longitud de la secuencia


    def standardize_dataset(self, 
                            full_processed_signals:list):
        # Find the max length of the signals
        max_length = max([len(x) for x in full_processed_signals])

        # Pad signal with zeros to normalize every signal to the same length
        signals_padded = [np.pad(x, (0, max_length - len(x))) for x in full_processed_signals]

        return signals_padded

    def pca_similarity(self, 
                       data:np.ndarray,
                       data_type:str):
        # Join signals in np.ndarray
        X = np.vstack(data)
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(data_scaled)   

        # Centroids of every class
        centroids = {}
        for label_idx in self.label_names_dict.keys():
            centroids[label_idx] = X_pca[np.array(self.labels) == label_idx].mean(axis=0)

        # Covariance matrix of PCA data
        cov_matrix = np.cov(X_pca.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        # Calculate distances between centroids
        distances = {}
        label_keys = list(centroids.keys())

        for i in range(len(label_keys)):
            for j in range(i + 1, len(label_keys)):
                label_i, label_j = label_keys[i], label_keys[j]

                # Euclidean distance
                euclidean_dist = np.linalg.norm(centroids[label_i] - centroids[label_j])

                # Mahalanobis distance
                mahal_dist = mahalanobis(centroids[label_i], centroids[label_j], inv_cov_matrix)

                distances[(label_i, label_j)] = {
                    "euclidean": euclidean_dist,
                    "mahalanobis": mahal_dist
                }

        print(f"\nCluster distance PCA - {data_type}:")
        for pair, dists in distances.items():
            label_names = [self.label_names_dict[pair[0]], self.label_names_dict[pair[1]]]
            print(f"Cluster {label_names[0]} vs {label_names[1]} -> Euclidean: {dists['euclidean']:.4f}, Mahalanobis: {dists['mahalanobis']:.4f}")

        return X_pca

    
    def tsne_similarity(self,
                        data:np.ndarray,
                        data_type:str):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
        X_tsne = tsne.fit_transform(data_scaled)
        
        # Covariance matrix of t-SNE data
        cov_matrix = np.cov(X_tsne.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        # Get centroids of every class
        centroids = {}
        for label_idx in self.label_names_dict.keys():
            centroids[label_idx] = X_tsne[np.array(self.labels) == label_idx].mean(axis=0)

        # Calculate distances between centroids
        distances = {}
        label_keys = list(centroids.keys())

        for i in range(len(label_keys)):
            for j in range(i + 1, len(label_keys)):
                label_i, label_j = label_keys[i], label_keys[j]

                # Euclidean distance
                euclidean_dist = np.linalg.norm(centroids[label_i] - centroids[label_j])

                # Mahalanobis distance
                mahal_dist = mahalanobis(centroids[label_i], centroids[label_j], inv_cov_matrix)

                distances[(label_i, label_j)] = {
                    "euclidean": euclidean_dist,
                    "mahalanobis": mahal_dist
                }

        print(f"\nCluster distance t-SNE - {data_type}:")
        for pair, dists in distances.items():
            label_names = [self.label_names_dict[pair[0]], self.label_names_dict[pair[1]]]
            print(f"Cluster {label_names[0]} vs {label_names[1]} -> Euclidean: {dists['euclidean']:.4f}, Mahalanobis: {dists['mahalanobis']:.4f}")

        return X_tsne


def load_dataset(config: dict) -> ConcatDataset:
    """
    Loads multiple datasets from the specified paths and concatenates them into a single dataset.

    Args:
        config (dict): Configuration dictionary containing dataset paths.

    Returns:
        ConcatDataset: A PyTorch dataset containing all loaded datasets.
    """
    datasets = [
        RealDataset(config=config, dataset_path=dataset_path, class_idx=class_idx)
        for class_idx, dataset_path in enumerate(config["dataset_paths"])
    ]

    return ConcatDataset(datasets)



def plot_pca_tsne_comparison(visualization_config: dict, 
                             sim_checker:SimilarityChecker):
    """
    Visualizes PCA (top row) and t-SNE (bottom row) projections of sequences and signals.

    This function generates a 2x2 grid of scatter plots:
    - Top row: PCA projections for sequences and signals.
    - Bottom row: t-SNE projections for sequences and signals.

    Parameters:
    - visualization_config (dict): Dictionary containing:
        - "X_pca" (list): PCA-transformed data as [X_pca_sequences, X_pca_signals].
        - "X_tsne" (list): t-SNE-transformed data as [X_tsne_sequences, X_tsne_signals].
        - "data_types" (list): Labels for datasets, e.g., ["Sequences", "Signals"].
    - sim_checker (SimilarityChecker): Instance containing labels and class information.

    Returns:
    - None (displays the plots).
    """

    # Extract PCA and t-SNE data
    data_types = visualization_config["data_types"]
    X_pca_sequences, X_pca_signals = visualization_config["X_pca"]
    X_tsne_sequences, X_tsne_signals = visualization_config["X_tsne"]

    # Create 2x2 figure layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Two rows, two columns

    # Define color mapping
    colors = ["red", "blue", "green"]

    # Titles for each subplot
    titles = [
        f"PCA - {data_types[0]}", f"PCA - {data_types[1]}",
        f"t-SNE - {data_types[0]}", f"t-SNE - {data_types[1]}"
    ]

    # List of datasets to plot
    transformed_data = [X_pca_sequences, X_pca_signals, X_tsne_sequences, X_tsne_signals]

    # Iterate over each subplot and plot the corresponding dataset
    for ax, X_data, title in zip(axes.flatten(), transformed_data, titles):
        for label_idx, color in zip(sim_checker.label_names_dict.keys(), colors):
            mask = np.array(sim_checker.labels) == label_idx
            ax.scatter(X_data[mask, 0], 
                       X_data[mask, 1], 
                       label=f"{sim_checker.label_names_dict[label_idx]}", 
                       alpha=0.6,
                       color=color)

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend()
        ax.set_title(title)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    config_path = args.config

    # Load all configurations from the training experiment file
    config = load_config(config_path)

    # Load and concatenate datasets
    full_dataset = load_dataset(config["dataset"])
    
    # Instantiate class of similaritychecker
    sim_checker = SimilarityChecker(full_dataset)

    # Do PCA
    X_sequence_pca = sim_checker.pca_similarity(data=sim_checker.kmer_frequencies, 
                                                data_type="Sequences")
    X_signal_pca = sim_checker.pca_similarity(data=sim_checker.std_signals,
                                              data_type="Signals")

    # Do T-SNE
    X_sequence_tsne = sim_checker.tsne_similarity(data=sim_checker.kmer_frequencies, 
                                                  data_type="Sequences")

    X_signal_tsne = sim_checker.tsne_similarity(data=sim_checker.std_signals, 
                                                  data_type="Signals")

    visualization_config = {
        "X_pca": [X_sequence_pca, X_signal_pca],
        "X_tsne": [X_sequence_tsne, X_signal_tsne],
        "data_types": ["Sequences", "Signals"]
    }

    # Plot PCA and t-SNE
    plot_pca_tsne_comparison(visualization_config=visualization_config,
                             sim_checker=sim_checker)