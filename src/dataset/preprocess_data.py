import pywt
import time
import numpy as np
from scipy.signal import butter, filtfilt, medfilt, fftconvolve
from src.dataset.domain.utils import generate_kmer_dict

class PreprocessData:
    def __init__(self,
                 rng:np.random.Generator,
                 step_ratio:float,
                 window_size:int,
                 kmers_size:int,
                 sampling_rate:int=4000,
                 cutoff:int = 1000,
                 median_kernel_size:int = 5):
        self.rng = rng

        self.step_ratio = step_ratio

        self.sampling_rate = sampling_rate

        self.kmers_size = kmers_size

        self.cutoff = cutoff

        self.window_size = window_size

        self.median_kernel_size = median_kernel_size

        self.kmer_dict = generate_kmer_dict(kmers_size=self.kmers_size)


    def lowpass_filter(self, signal: np.ndarray, order: int = 4) -> np.ndarray:
        """
        Applies a low-pass Butterworth filter to the input signal.

        This function filters the signal to remove high-frequency components while preserving 
        lower-frequency content. The cutoff frequency is normalized with respect to the 
        Nyquist frequency to ensure stability.

        Parameters:
        signal (np.ndarray): The input signal array to be filtered.
        order (int, optional): The order of the Butterworth filter. Default is 4.

        Returns:
        np.ndarray: The filtered signal.

        Raises:
        ValueError: If the cutoff frequency is not properly defined.
        """

        if not hasattr(self, "sampling_rate") or not hasattr(self, "cutoff"):
            raise ValueError("Both 'sampling_rate' and 'cutoff' must be defined in the class.")

        # Compute Nyquist frequency
        nyquist = 0.5 * self.sampling_rate

        # Normalize cutoff frequency and ensure it is within a valid range
        normal_cutoff = min(self.cutoff / nyquist, 0.99)

        if normal_cutoff <= 0:
            raise ValueError("Cutoff frequency must be greater than zero and within a valid range.")

        # Design Butterworth low-pass filter
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        # Apply the filter using zero-phase filtering to avoid phase distortion
        return filtfilt(b, a, signal)


    def wavelet_transform(self, 
                        signal: np.ndarray, 
                        wavelet: str = 'db4', 
                        level: int = 5, 
                        threshold: float = 0.1) -> np.ndarray:
        """
        Applies a wavelet transform to the given signal, performs thresholding, and reconstructs the filtered signal.

        This function uses the Discrete Wavelet Transform (DWT) to decompose the signal into different frequency components. 
        Small coefficients are thresholded to remove noise, and the signal is then reconstructed.

        Parameters:
        signal (np.ndarray): The input signal array.
        wavelet (str, optional): The wavelet type to use for decomposition. Default is 'db4'.
        level (int, optional): The number of decomposition levels. Default is 5.
        threshold (float, optional): The threshold for coefficient shrinking. Default is 0.1.

        Returns:
        np.ndarray: The filtered signal after wavelet transform and reconstruction.

        Raises:
        ValueError: If the signal is empty.
        """

        if signal.size == 0:
            raise ValueError("Input signal cannot be empty.")

        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        # Apply thresholding to remove small coefficients (noise reduction)
        coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

        # Reconstruct the signal using thresholded coefficients
        filtered_signal = pywt.waverec(coeffs, wavelet)

        return filtered_signal
    

    def modified_zscore(self, signal: np.ndarray) -> np.ndarray:
        """
        Computes the modified Z-score for a given signal and replaces outliers.

        This function uses the median and the Median Absolute Deviation (MAD) 
        for robust outlier detection and handles them efficiently.

        Parameters:
            signal (np.ndarray): The input signal array.

        Returns:
            np.ndarray: An array of modified Z-scores.

        Raises:
            ValueError: If the input signal is empty.
        """
        if signal.size == 0:
            raise ValueError("Input signal cannot be empty.")

        # Compute median and MAD (Median Absolute Deviation)
        median_X = np.median(signal)
        MAD = np.median(np.abs(signal - median_X))

        # Prevent division by zero
        MAD = max(MAD, 1e-9)

        # Compute modified Z-scores
        z_scores = (signal - median_X) / MAD

        # Outlier detection
        threshold = 3.5  # Outlier detection threshold
        outliers = np.abs(z_scores) > threshold

        if np.any(outliers):  # Only proceed if there are outliers
            # Get indices of outliers
            outlier_indices = np.where(outliers)[0]

            # Create a copy of the signal to avoid modifying the original during indexing
            signal_copy = signal.copy()

            # Get valid indices (excluding first and last since they can't be replaced using neighbors)
            valid_indices = outlier_indices[(outlier_indices > 0) & (outlier_indices < len(signal) - 1)]

            # Vectorized replacement: Replace outliers with the mean of adjacent values
            signal_copy[valid_indices] = (signal[valid_indices - 1] + signal[valid_indices + 1]) / 2

            return signal_copy  # Return the modified signal

        return signal  # If no outliers, return the original signal




    def apply_sliding_window(self, signal: np.ndarray) -> np.ndarray:
        """
        Applies a sliding window to the given signal.

        This function segments the input signal into overlapping windows of a specified size,
        moving with a step determined by the `step_ratio`. The step size is calculated as 
        `window_size * step_ratio`, ensuring controlled overlap between windows.

        Parameters:
        signal (np.ndarray): The input signal array.

        Returns:
        np.ndarray: A 2D NumPy array where each row represents a windowed segment of the signal.

        Raises:
        ValueError: If `window_size` is not within a valid range (greater than 0 and 
                    less than or equal to the signal length).
        ValueError: If `step_ratio` is not within the valid range (between 0 and 1).
        """
        
        # Validate window size
        if not (0 < self.window_size <= len(signal)):
            raise ValueError("Window size must be greater than 0 and less than or equal to the signal length.")
        
        # Validate step ratio
        if not (0 < self.step_ratio <= 1):
            raise ValueError("Step ratio must be between 0 and 1.")
        
        # Compute step size and number of windows
        step = max(1, int(self.window_size * self.step_ratio))
        window_indices = range(0, len(signal) - self.window_size + 1, step)

        # Generate sliding windows efficiently
        window_signal = [signal[i : i + self.window_size] for i in window_indices]
        
        return np.array(window_signal, dtype=np.float32)


    def preprocess_signal(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses the input signal by applying multiple filtering and normalization steps.

        This function enhances the signal quality by removing noise, smoothing, normalizing, 
        and segmenting it into overlapping windows.

        Processing Steps:
        1. **Low-pass filter**: Removes high-frequency noise.
        2. **Median filter**: Eliminates impulsive noise and sharp peaks.
        3. **Moving average smoothing**: Smooths the signal further.
        4. **Wavelet transform**: Performs noise reduction using wavelet decomposition.
        5. **Modified Z-score normalization**: Normalizes the signal for outlier handling.
        6. **Sliding window segmentation**: Segments the signal into overlapping windows.

        Parameters:
        signal (np.ndarray): The input signal array.

        Returns:
        tuple[np.ndarray, np.ndarray]: 
            - The fully preprocessed signal (np.float32).
            - The windowed segments of the preprocessed signal.
        
        Raises:
        ValueError: If required attributes (`median_kernel_size`, `window_size`) are missing.
        """

        if not hasattr(self, "median_kernel_size") or not hasattr(self, "window_size"):
            raise ValueError("Attributes 'median_kernel_size' and 'window_size' must be defined in the class.")

        start = time.time()
        # Step 1: Apply low-pass filter to remove high-frequency noise
        preprocessed_signal = self.lowpass_filter(signal)
        # print(f"Step 1: {time.time() - start:.4f} secs")

        start = time.time()
        # Step 2: Apply median filter to remove impulsive noise
        preprocessed_signal = medfilt(preprocessed_signal, self.median_kernel_size)
        # print(f"Step 2: {time.time() - start:.4f} secs")

        start = time.time()
       
        # # Step 3: Apply moving average smoothing
        # preprocessed_signal = np.convolve(
        #     preprocessed_signal, 
        #     np.ones(self.window_size) / self.window_size, 
        #     mode="same"
        # )
        
        # Step 3: Apply moving average smoothing
        # Faster moving average using FFT-based convolution
        preprocessed_signal = fftconvolve(
            preprocessed_signal, 
            np.ones(self.window_size) / self.window_size, 
            mode="same"
        )
        # print(f"Step 3: {time.time() - start:.4f} secs")

        start = time.time()
        # Step 4: Apply wavelet transform for noise reduction
        preprocessed_signal = self.wavelet_transform(preprocessed_signal)
        # print(f"Step 4: {time.time() - start:.4f} secs")

        start = time.time()
        # Step 5: Normalize using the modified Z-score
        preprocessed_signal = np.array(self.modified_zscore(preprocessed_signal), dtype=np.float32)
        # print(f"Step 5: {time.time() - start:.4f} secs")

        start = time.time()
        # Step 6: Apply sliding window segmentation
        window_signal = self.apply_sliding_window(signal=preprocessed_signal)
        # print(f"Step 6: {time.time() - start:.4f} secs")

        return preprocessed_signal, window_signal


    def sequence_to_kmer_indices(self, sequence: str) -> list[int]:
        """
        Converts a DNA sequence into a list of K-mer indices.

        This function extracts overlapping K-mers from the input DNA sequence and maps them 
        to their corresponding indices using a predefined K-mer dictionary.

        Parameters:
        sequence (str): The input DNA sequence (e.g., "ACGTAG").

        Returns:
        list[int]: A list of indices representing the K-mers in the sequence.

        Raises:
        ValueError: If a K-mer is not found in the dictionary.
        AttributeError: If required attributes (`k_mers_size` or `kmer_dict`) are missing.
        """

        if not hasattr(self, "kmers_size") or not hasattr(self, "kmer_dict"):
            raise AttributeError("Attributes 'kmers_size' and 'kmer_dict' must be defined in the class.")

        kmer_indices = []
        sequence_length = len(sequence)

        # Iterate over the sequence to extract K-mers
        for i in range(sequence_length - self.kmers_size + 1):
            kmer = sequence[i : i + self.kmers_size]

            # Check if the K-mer exists in the dictionary
            if kmer in self.kmer_dict:
                kmer_indices.append(self.kmer_dict[kmer])
            else:
                raise ValueError(f"K-mer '{kmer}' not found. Consider changing the K-mer size.")

        return kmer_indices

