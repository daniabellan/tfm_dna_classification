#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <species_name>"
    exit 1
fi

# Activate the Conda environment
CONDA_ENV="tfm"
echo "=== Activating Conda environment '$CONDA_ENV' ==="
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate Conda environment '$CONDA_ENV'"
    exit 1
fi

# Assign species name to a variable
SPECIE=$1

# Define paths
BASE_DIR="data/$SPECIE"
BAM_FILE="$BASE_DIR/basecalling_output/basecalled_signal.sam"
FASTA_DIR="$BASE_DIR/basecalling_fasta"
FASTA_OUTPUT="$FASTA_DIR/basecalled_signal.fa"
GENOME_DIR="$BASE_DIR/genome_reference"
JELLYFISH_DIR="$BASE_DIR/jellyfish_output"

# Create necessary directories
mkdir -p "$FASTA_DIR"
mkdir -p "$JELLYFISH_DIR"

# Function to run Jellyfish and dump results
run_jellyfish() {
    local input_file=$1
    local output_prefix=$2
    local jellyfish_output="$JELLYFISH_DIR/${output_prefix}.jf"
    local jellyfish_dump_output="$JELLYFISH_DIR/${output_prefix}.txt"
    
    echo "Running Jellyfish on: $input_file"
    jellyfish count -m 3 -s 100M -t 8 -C -o "$jellyfish_output" "$input_file"
    echo "Jellyfish output saved at: $jellyfish_output"
    
    echo "Dumping Jellyfish results..."
    jellyfish dump -c -t -o "$jellyfish_dump_output" "$jellyfish_output"
    echo "Jellyfish dump saved at: $jellyfish_dump_output"
}

# Step 1: Convert BAM to FASTA
if [ -f "$BAM_FILE" ]; then
    echo "Processing BAM file: $BAM_FILE"
    samtools bam2fq "$BAM_FILE" | seqtk seq -A > "$FASTA_OUTPUT"
    echo "FASTA file saved at: $FASTA_OUTPUT"
    
    # Run Jellyfish on the generated FASTA
    run_jellyfish "$FASTA_OUTPUT" "${SPECIE}_bam"
else
    echo "Error: BAM file not found at $BAM_FILE"
    exit 1
fi

# Step 2: Find genome reference file (.fna or .fa)
GENOME_FILE=$(find "$GENOME_DIR" -type f \( -name "*.fna" -o -name "*.fa" \) | head -n 1)

if [ -z "$GENOME_FILE" ]; then
    echo "Error: No genome reference file found in $GENOME_DIR"
    exit 1
fi

# Run Jellyfish on the genome reference file
run_jellyfish "$GENOME_FILE" "${SPECIE}_genome"

echo "Process completed successfully."
