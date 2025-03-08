#!/bin/bash

# Activate Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Verify that the Conda environment was activated successfully
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to activate Conda environment '$CONDA_ENV'."
    exit 1
fi

# Define the base directory containing species data
BASE_DIR="data"

# Find all genome reference files (.fa or .fna) within subdirectories
GENOMES=($(find "$BASE_DIR" -type f \( -name "*.fa" -o -name "*.fna" \)))

# Define the output directory for FastANI results
OUTPUT_DIR="fastani_results"
mkdir -p "$OUTPUT_DIR"

# Define the output CSV file
OUTPUT_FILE="$OUTPUT_DIR/ANI_results.csv"
echo "Query,Reference,ANI" > "$OUTPUT_FILE"

# Define the error log file
ERROR_LOG="$OUTPUT_DIR/error_log.txt"
> "$ERROR_LOG"  # Clear the file before starting

# Loop through all genome pairs and compare them using FastANI
for query in "${GENOMES[@]}"; do
    for reference in "${GENOMES[@]}"; do
        if [[ "$query" != "$reference" ]]; then
            echo "🔍 Comparing: $query vs $reference"

            # Run FastANI and capture output
            fastANI -q "$query" -r "$reference" --output "$OUTPUT_DIR/temp_output.txt" 2>> "$ERROR_LOG"

            # If FastANI execution is successful, extract the ANI value
            if [[ -s "$OUTPUT_DIR/temp_output.txt" ]]; then
                ANI_VALUE=$(awk '{print $3}' "$OUTPUT_DIR/temp_output.txt")
                echo "$query,$reference,$ANI_VALUE" >> "$OUTPUT_FILE"
            else
                echo "$query,$reference,ERROR" >> "$OUTPUT_FILE"
            fi
        fi
    done
done

# Remove temporary output file
rm -f "$OUTPUT_DIR/temp_output.txt"

echo "Comparison completed. Results saved in $OUTPUT_FILE"
echo "  If errors occurred, check $ERROR_LOG"