from torch.utils.data import Dataset
import numpy as np
import time
from typing import List, Dict, Union

from src.dataset.domain.dataclass import SignalSequenceData
from src.dataset.preprocess_data import PreprocessData
from src.dataset.load_h5_file import load_h5_file

class RealDataset(Dataset):
    """
    A dataset class for genomic sequence processing.

    This class loads genomic sequences from an HDF5 file, applies preprocessing steps 
    (e.g., filtering, k-mer extraction, segmentation), and provides an interface 
    compatible with PyTorch's Dataset and ConcatDataset.

    Attributes:
    dataset_path (str): Path to the dataset file.
    class_idx (int): Index of the class to filter from the dataset.
    rng (np.random.Generator): Random number generator for reproducibility.
    preprocess (bool): Flag to determine if preprocessing should be applied.
    window_size (int): Size of the sliding window for signal segmentation.
    step_ratio (float): Ratio determining step size in sliding window segmentation.
    num_samples (int): Number of samples to load from the dataset.
    kmers_size (int): Size of k-mers to generate.
    data (List[SignalSequenceData]): List storing processed dataset samples.

    Methods:
    __init__(config: dict, dataset_path: str, class_idx: int, seed: int = 42)
        Initializes the dataset by loading raw data and applying preprocessing.
    __len__() -> int
        Returns the total number of samples in the dataset.
    __getitem__(index: int) -> Dict[str, Union[np.ndarray, int]]
        Retrieves a sample as a dictionary for PyTorch DataLoader compatibility.
    """

    def __init__(self, config: dict, dataset_path: str, class_idx: int, seed: int = 42):
        """
        Initializes the RealDataset class by loading genomic data and optionally 
        applying preprocessing steps.

        Parameters:
        config (dict): Configuration dictionary with preprocessing parameters.
        dataset_path (str): Path to the dataset file.
        class_idx (int): Class index to filter the dataset.
        seed (int, optional): Random seed for reproducibility. Default is 42.
        """

        self.data: List[SignalSequenceData] = []

        self.dataset_path = dataset_path
        self.class_idx = class_idx
        self.rng = np.random.default_rng(seed)

        # Extract configuration parameters
        self.preprocess = config.get("preprocess", True)
        self.window_size = config.get("window_size", 1500)
        self.step_ratio = config.get("step_ratio", 0.5)
        self.num_samples = config.get("num_samples", 100)
        self.kmers_size = config.get("kmers_size", 3)

        # Load raw dataset
        dataset = load_h5_file(dataset_path=self.dataset_path,
                               num_samples=self.num_samples,
                               class_idx=self.class_idx,
                               rng=self.rng)

        # Apply preprocessing if enabled
        if self.preprocess:
            prep_data = PreprocessData(
                rng=self.rng,
                step_ratio=self.step_ratio,
                window_size=self.window_size,
                kmers_size=self.kmers_size
            )

            start_time = time.time()
            for read in dataset:
                # Preprocess signal and create windowed segments
                processed_signal, window_signal = prep_data.preprocess_signal(read.signal_pa)

                start = time.time()
                # Convert sequence to k-mer indices
                kmers = prep_data.sequence_to_kmer_indices(read.sequence)
                # print(f"sequence_to_kmer_indices: {time.time() - start:.4f} secs")

                # Store processed data
                self.data.append(SignalSequenceData(
                    signal_pa=read.signal_pa,
                    sequence=read.sequence,
                    label=read.label,
                    full_processed_signal=processed_signal,
                    window_signal=window_signal,
                    kmers=kmers
                ))

            elapsed_time = time.time() - start_time
            print(f"Preprocessing completed in {elapsed_time:.4f} seconds.")
        else:
            # Store raw data if preprocessing is disabled
            self.data = dataset


    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
        int: The number of samples.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, int]]:
        """
        Retrieves a sample at the specified index in dictionary format.

        Parameters:
        index (int): Index of the sample to retrieve.

        Returns:
        Dict[str, Union[np.ndarray, int]]: A dictionary containing:
            - "signal_pa" (np.ndarray): Raw signal data.
            - "full_processed_signal" (np.ndarray): Fully processed signal.
            - "window_signal" (np.ndarray): Windowed signal segments.
            - "kmers" (np.ndarray): K-mer indices.
            - "label" (int): Class label of the sample.
        """

        sample = self.data[index]

        return {
            "signal_pa": sample.signal_pa,
            "full_processed_signal": sample.full_processed_signal,
            "window_signal": sample.window_signal,
            "kmers": sample.kmers,
            "label": sample.label
        }
