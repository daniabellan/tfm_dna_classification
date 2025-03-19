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
from src.dataset.domain.kmer_utils import generate_kmer_dict

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)


class SimilarityChecker:
    def __init__(self, loaded_dataset: RealDataset):
        """
        Initialize the SimilarityChecker with a loaded dataset.

        Args:
            loaded_dataset (RealDataset): The loaded dataset.
        """
        # Use the random generator from the first dataset
        self.rng = loaded_dataset.datasets[0].rng

        # Retrieve k-mer size from the dataset
        self.kmer_size = loaded_dataset.datasets[0].kmers_size

        # Extract class names and labels for each signal in the dataset
        self.labels = []
        self.sequences = []
        self.label_names_dict = {}
        for dataset in loaded_dataset.datasets:
            real_name = dataset.dataset_path.split("/")[-3]
            # Map dataset folder names to human-readable labels
            if real_name == "ecoli_k12_real":
                label_name = "E. Coli K12"
            elif real_name == "ecoli_O104H4":
                label_name = "E. Coli O104H4"
            elif real_name == "ecoli_O157H7":
                label_name = "E. Coli O157H7"
            elif real_name == "salmonella_enterica":
                label_name = "Salmonella Enterica"
            elif real_name == "mm39":
                label_name = real_name
            else:
                label_name = real_name

            self.label_names_dict[dataset.class_idx] = label_name
            for signal in dataset.data:
                self.labels.append(signal.label)
                self.sequences.append(signal.sequence)

        # Number of classes
        self.num_classes = len(self.label_names_dict)

        # Extract and standardize signals (pre-processed signals)
        full_processed_signals = []
        for dataset in loaded_dataset:
            full_processed_signals.append(dataset["full_processed_signal"])
        self.std_signals = self.standardize_dataset(full_processed_signals)

        # Create k-mer frequency vectors for each sequence
        self.kmer_frequencies = np.array([self.compute_kmer_frequencies(seq) for seq in self.sequences])

    def compute_kmer_frequencies(self, sequence: str):
        """
        Compute the normalized k-mer frequency vector for a given sequence.

        Args:
            sequence (str): The input sequence.

        Returns:
            numpy.ndarray: Normalized frequency vector of k-mers.
        """
        # Generate a dictionary mapping k-mers to their indices
        kmer_dict = generate_kmer_dict(self.kmer_size)
        # Initialize a frequency vector with zeros
        vector = np.zeros(len(kmer_dict))
        for i in range(len(sequence) - self.kmer_size + 1):
            kmer = sequence[i:i + self.kmer_size]
            if kmer in kmer_dict:
                vector[kmer_dict[kmer]] += 1  # Increase count for the k-mer
        return vector / np.sum(vector)  # Normalize by the total count

    def standardize_dataset(self, full_processed_signals: list):
        """
        Standardize signals by padding them with zeros so that all signals have the same length.

        Args:
            full_processed_signals (list): List of signals as numpy arrays.

        Returns:
            list: List of padded signals.
        """
        # Find the maximum signal length
        max_length = max([len(x) for x in full_processed_signals])
        # Pad each signal with zeros to match the maximum length
        signals_padded = [np.pad(x, (0, max_length - len(x))) for x in full_processed_signals]
        return signals_padded

    def pca_similarity(self, data: np.ndarray, data_type: str):
        """
        Apply PCA on the data and print cluster distances.

        Args:
            data (np.ndarray): Data to project using PCA.
            data_type (str): A label for the data type (e.g., "Sequences" or "Signals").

        Returns:
            np.ndarray: PCA-transformed data.
        """
        # Stack the list of arrays into a single numpy array
        X = np.vstack(data)
        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(X)
        # Perform PCA to reduce dimensions to 2
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(data_scaled)

        # Compute centroids for each class
        centroids = {}
        for label_idx in self.label_names_dict.keys():
            centroids[label_idx] = X_pca[np.array(self.labels) == label_idx].mean(axis=0)

        # Compute covariance matrix and its inverse for Mahalanobis distance
        cov_matrix = np.cov(X_pca.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        # Calculate Euclidean and Mahalanobis distances between centroids
        distances = {}
        label_keys = list(centroids.keys())
        for i in range(len(label_keys)):
            for j in range(i + 1, len(label_keys)):
                label_i, label_j = label_keys[i], label_keys[j]
                euclidean_dist = np.linalg.norm(centroids[label_i] - centroids[label_j])
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

    def tsne_similarity(self, data: np.ndarray, data_type: str):
        """
        Apply t-SNE on the data and print cluster distances.

        Args:
            data (np.ndarray): Data to project using t-SNE.
            data_type (str): A label for the data type (e.g., "Sequences" or "Signals").

        Returns:
            np.ndarray: t-SNE-transformed data.
        """
        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        # Perform t-SNE to reduce dimensions to 2
        tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
        X_tsne = tsne.fit_transform(data_scaled)

        # Compute centroids for each class in t-SNE space
        centroids = {}
        for label_idx in self.label_names_dict.keys():
            centroids[label_idx] = X_tsne[np.array(self.labels) == label_idx].mean(axis=0)

        # Compute covariance matrix and its inverse for Mahalanobis distance
        cov_matrix = np.cov(X_tsne.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        # Calculate Euclidean and Mahalanobis distances between centroids
        distances = {}
        label_keys = list(centroids.keys())
        for i in range(len(label_keys)):
            for j in range(i + 1, len(label_keys)):
                label_i, label_j = label_keys[i], label_keys[j]
                euclidean_dist = np.linalg.norm(centroids[label_i] - centroids[label_j])
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
    Load multiple datasets from specified paths and concatenate them.

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


def plot_pca_tsne_separately(visualization_config: dict, sim_checker: SimilarityChecker):
    """
    Visualize PCA and t-SNE projections separately for K-mers (Sequences) and Signals.
    
    Two separate figures are generated:
      - One for K-mers (Sequences) with a 1x2 layout (PCA and t-SNE).
      - One for Signals with a 1x2 layout (PCA and t-SNE).

    Args:
        visualization_config (dict): Dictionary containing:
            - "X_pca" (list): PCA-transformed data as [X_pca_sequences, X_pca_signals].
            - "X_tsne" (list): t-SNE-transformed data as [X_tsne_sequences, X_tsne_signals].
            - "data_types" (list): Labels for data types, e.g., ["Sequences", "Signals"].
        sim_checker (SimilarityChecker): Instance containing label and class information.
    """
    colors = ["red", "blue", "green"]

    # Plot for K-mers (Sequences)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = [f"PCA - {visualization_config['data_types'][0]}", f"t-SNE - {visualization_config['data_types'][0]}"]
    datasets_to_plot = [visualization_config["X_pca"][0], visualization_config["X_tsne"][0]]
    
    for ax, data, title in zip(axes, datasets_to_plot, titles):
        for label_idx, color in zip(sim_checker.label_names_dict.keys(), colors):
            mask = np.array(sim_checker.labels) == label_idx
            ax.scatter(data[mask, 0], data[mask, 1],
                       label=sim_checker.label_names_dict[label_idx],
                       color=color, alpha=0.6)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend()
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    # Plot for Signals
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = [f"PCA - {visualization_config['data_types'][1]}", f"t-SNE - {visualization_config['data_types'][1]}"]
    datasets_to_plot = [visualization_config["X_pca"][1], visualization_config["X_tsne"][1]]
    
    for ax, data, title in zip(axes, datasets_to_plot, titles):
        for label_idx, color in zip(sim_checker.label_names_dict.keys(), colors):
            mask = np.array(sim_checker.labels) == label_idx
            ax.scatter(data[mask, 0], data[mask, 1],
                       label=sim_checker.label_names_dict[label_idx],
                       color=color, alpha=0.6)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend()
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    pass

if __name__ == "__main__":
    # Parse command-line arguments to get configuration file path
    args = parse_args()
    config_path = args.config

    # Load configuration
    config = load_config(config_path)

    # Modify the number of samples for simplicity in visualization
    config["dataset"]["num_samples"] = 200

    # Load and concatenate datasets
    full_dataset = load_dataset(config["dataset"])

    # Create an instance of SimilarityChecker
    sim_checker = SimilarityChecker(full_dataset)

    # Compute PCA and t-SNE projections for K-mers (Sequences)
    X_sequence_pca = sim_checker.pca_similarity(data=sim_checker.kmer_frequencies, data_type="Sequences")
    X_sequence_tsne = sim_checker.tsne_similarity(data=sim_checker.kmer_frequencies, data_type="Sequences")

    # Compute PCA and t-SNE projections for Signals
    X_signal_pca = sim_checker.pca_similarity(data=sim_checker.std_signals, data_type="Signals")
    X_signal_tsne = sim_checker.tsne_similarity(data=sim_checker.std_signals, data_type="Signals")

    visualization_config = {
        "X_pca": [X_sequence_pca, X_signal_pca],
        "X_tsne": [X_sequence_tsne, X_signal_tsne],
        "data_types": ["Sequences", "Signals"]
    }

    # Plot PCA and t-SNE projections separately for K-mers and Signals
    plot_pca_tsne_separately(visualization_config=visualization_config, sim_checker=sim_checker)
