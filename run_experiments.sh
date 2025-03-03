#!/bin/bash

# Nombre del entorno de Conda
CONDA_ENV="tfm"
# Script Python que entrena la red neuronal
TRAIN_SCRIPT="main.py"
# Carpeta donde están los archivos de configuración
CONFIG_DIR="configs/experiment_configs"

# Lista de archivos específicos a ejecutar
CONFIG_FILES=(
    "9000s_km3_k12_O104_0157_real.yaml"
)

# Activar Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Verificar que el entorno se activó correctamente
if [[ $? -ne 0 ]]; then
    echo "ERROR: No se pudo activar el entorno de Conda '$CONDA_ENV'."
    exit 1
fi

# Ejecutar cada experimento en segundo plano con nohup
for CONFIG in "${CONFIG_FILES[@]}"; do
    CONFIG_PATH="$CONFIG_DIR/$CONFIG"

    # Verificar si el archivo existe antes de ejecutarlo
    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "WARNING: Archivo de configuración '$CONFIG_PATH' no encontrado. Saltando..."
        continue
    fi

    # Obtener timestamp en formato: YYYY-MM-DDTHH-MM-SS
    TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")

    # Crear la carpeta de logs específica para este archivo de configuración
    BASE_NAME=$(basename "$CONFIG" .yaml)
    LOG_DIR="logs/${BASE_NAME}"
    mkdir -p "$LOG_DIR"

    # Archivo de log dentro de la subcarpeta
    LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

    echo "Ejecutando experimento con configuración: $CONFIG_PATH"
    echo "Archivo de log: $LOG_FILE"

    # Lanzar el script en segundo plano con nohup
    nohup python -u $TRAIN_SCRIPT --config "$CONFIG_PATH" > "$LOG_FILE" 2>&1 &

    echo "Experimento con $CONFIG_PATH en ejecución. Revisa 'tail -f $LOG_FILE' para ver el progreso."
done

echo "Todos los experimentos han sido lanzados en segundo plano."
