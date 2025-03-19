#!/bin/bash

# This script processes genomic data for a given species.
# It verifies the presence of a genome reference file, generates artificial signal data,
# converts formats, performs basecalling, and preprocesses the data for further analysis.

# Ensure that a species name is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <species_name>"
    echo "A reference genome file must exist in: <species_name>/genome_reference/ with .fa or .fna extension"
    exit 1
fi

# Define variables
SPECIES=$1
BASE_DIR="data/${SPECIES}"
GENOME_DIR="${BASE_DIR}/genome_reference"
CONDA_ENV="tfm"

# Locate the genome reference file (.fa or .fna)
GENOME_REF=$(find "$GENOME_DIR" -maxdepth 1 -type f \( -name "*.fa" -o -name "*.fna" \) | head -n 1)

# Check if a valid genome reference file was found
if [ -z "$GENOME_REF" ]; then
    echo "Error: No reference genome found in $GENOME_DIR with .fa or .fna extension"
    exit 1
fi

echo "=== Using genome reference: $GENOME_REF ==="

# Create necessary directories
mkdir -p "${BASE_DIR}/blow5"                # Artificially generated data using Squigulator
mkdir -p "${BASE_DIR}/fast5"                # Transformed data from BLOW5 to FAST5
mkdir -p "${BASE_DIR}/basecalling_output"   # Dorado basecalling output
mkdir -p "${BASE_DIR}/final_data"           # H5 file for training/testing

# Activate the Conda environment
echo "=== Activating Conda environment '$CONDA_ENV' ==="
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate Conda environment '$CONDA_ENV'"
    exit 1
fi

# Generate electric signal using Squigulator
echo "=== Generating electric signal with Squigulator ==="
if [ $SPECIES = "mm39" ]; then
    FOLD_COVERAGE=1
else
    FOLD_COVERAGE=30
fi

squigulator "$GENOME_REF" -x dna-r9-min -o "${BASE_DIR}/blow5/reads.blow5" \
    --seed 42 -f $FOLD_COVERAGE -r 12000 --prefix=yes --bps 400 --verbose 4 -t 16
if [ $? -ne 0 ]; then
    echo "Error: Squigulator execution failed"
    exit 1
fi

# Convert BLOW5 format to FAST5 using slow5tools
echo "=== Converting BLOW5 to FAST5 with slow5tools ==="
slow5tools s2f "${BASE_DIR}/blow5/" -d "${BASE_DIR}/fast5/"
if [ $? -ne 0 ]; then
    echo "Error: slow5tools execution failed"
    exit 1
fi

# Perform basecalling with Dorado
echo "=== Performing basecalling with Dorado ==="
dorado basecaller --emit-sam -b 768 dorado_models/dna_r9.4.1_e8_sup@v3.6 "${BASE_DIR}/fast5/" -v > \
    "${BASE_DIR}/basecalling_output/basecalled_signal.sam"
if [ $? -ne 0 ]; then
    echo "Error: Dorado execution failed"
    exit 1
fi

# Preprocess data using a Python script
echo "=== Running preprocessing with Python ==="
python synthetic_dataset_functions/nanopore_data_merger.py \
    "$BASE_DIR/basecalling_output/basecalled_signal.sam" \
    "$BASE_DIR/fast5" \
    "$BASE_DIR/final_data/matched_data.h5" --verbose
if [ $? -ne 0 ]; then
    echo "Error: Python preprocessing script execution failed"
    exit 1
fi

echo "=== Process completed successfully ==="