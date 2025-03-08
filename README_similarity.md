# PCA and t-SNE Similarity Analysis

## Overview
This script performs similarity analysis using **Principal Component Analysis (PCA)** and **t-Distributed Stochastic Neighbor Embedding (t-SNE)** on genomic sequences and their corresponding signals. The analysis allows for visualizing clustering patterns and measuring distances between data classes using **Euclidean** and **Mahalanobis distances**.

## Workflow

### 1. **Loading the Dataset**
- The script loads multiple datasets specified in the configuration file.
- Extracts class labels and sequences/signals.

### 2. **Preprocessing**
- **K-mer Frequency Calculation**: Converts sequence data into k-mer frequency vectors.
- **Signal Standardization**: Pads signal data to the maximum sequence length and normalizes them.

### 3. **PCA Analysis**
- Standardizes input data.
- Reduces dimensionality to 2D using PCA.
- Computes **Euclidean** and **Mahalanobis distances** between class centroids.

### 4. **t-SNE Analysis**
- Standardizes input data.
- Reduces dimensionality to 2D using t-SNE.
- Computes **Euclidean** and **Mahalanobis distances** between class centroids.

### 5. **Visualization**
- Generates **scatter plots** comparing PCA and t-SNE projections.
- Uses different colors to represent different classes.



# Genome Reference Similarity Analysis using FastANI

## Overview
This script automates the process of calculating **Average Nucleotide Identity (ANI)** between genome reference files using **FastANI**. The script scans for genome reference files in a specified directory, performs pairwise ANI comparisons, and logs the results in a structured format.

### Dataset Structure
The script expects genome reference files in **FASTA** format (`.fa` or `.fna`) stored inside the `data/` directory or its subdirectories. 

## Workflow
### 1. **Activate Conda Environment**
The script starts by activating the Conda environment where **FastANI** is installed.

### 2. **Identify Genome Reference Files**
It scans for all genome reference files (`.fa`, `.fna`) within `data/` and stores them in a list.

### 3. **Pairwise Comparisons**
- Each genome file is compared **against all other genome files**.
- **FastANI** computes the ANI value between each pair.
- The computed ANI values are stored in a CSV file.

### 4. **Error Handling**
- If FastANI fails, an **error log** is created.
- If no ANI value is obtained, an `ERROR` entry is logged in the CSV file.

### 5. **Results Storage**
- ANI results are saved in `fastani_results/ANI_results.csv`
- Errors (if any) are logged in `fastani_results/error_log.txt`