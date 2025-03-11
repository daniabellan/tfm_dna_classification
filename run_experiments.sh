#!/bin/bash

# =============================
# Experiment Execution Script
# =============================

# Define paths
PROJECT_ROOT="$PWD"  # Get the current project root directory
CONDA_ENV="tfm"
TRAIN_SCRIPT="src/main.py"
CONFIG_DIR="src/configs/experiment"

# List of configuration files
CONFIG_FILES=(
    "experiment_1_3.yaml"
)

# =============================
# Activate Conda Environment
# =============================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

if [[ $? -ne 0 ]]; then
    echo "ERROR: Could not activate Conda environment '$CONDA_ENV'."
    exit 1
fi

# =============================
# Execute Each Experiment
# =============================

for CONFIG in "${CONFIG_FILES[@]}"; do
    CONFIG_PATH="$PROJECT_ROOT/$CONFIG_DIR/$CONFIG"

    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "WARNING: Configuration file '$CONFIG_PATH' not found. Skipping..."
        continue
    fi

    TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
    BASE_NAME=$(basename "$CONFIG" .yaml)
    LOG_DIR="$PROJECT_ROOT/logs/${BASE_NAME}"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

    echo "Starting experiment with configuration: $CONFIG_PATH"
    echo "Log file: $LOG_FILE"

    # =============================
    # Set PYTHONPATH and Run Experiment
    # =============================

    export PYTHONPATH="$PROJECT_ROOT"
    cd "$PROJECT_ROOT" || exit 1  # Ensure we are in the project root

    nohup python -u "$TRAIN_SCRIPT" --config "$CONFIG_PATH" > "$LOG_FILE" 2>&1 &

    echo "Experiment with $CONFIG is running. Check logs with: tail -f $LOG_FILE"
done

echo "All experiments have been launched in the background."
