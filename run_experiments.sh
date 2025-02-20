#!/bin/bash

# Nombre del entorno de Conda
CONDA_ENV="tfm"
# Script Python que entrena la red neuronal
TRAIN_SCRIPT="main.py"
# Carpeta donde están los archivos de configuración
CONFIG_DIR="configs/experiment_configs"

# Lista de archivos específicos a ejecutar
CONFIG_FILES=(
    "7000s_k3_ec_salm_mm39_signal.yaml"
)

# Activar Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Verificar que el entorno se activó correctamente
if [[ $? -ne 0 ]]; then
    echo "Error: No se pudo activar el entorno de Conda '$CONDA_ENV'."
    exit 1
fi

# Ejecutar cada experimento en segundo plano con nohup
for CONFIG in "${CONFIG_FILES[@]}"; do
    CONFIG_PATH="$CONFIG_DIR/$CONFIG"

    # Verificar si el archivo existe antes de ejecutarlo
    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "⚠️ Advertencia: Archivo de configuración '$CONFIG_PATH' no encontrado. Saltando..."
        continue
    fi

    # Obtener timestamp en formato legible: YYYY-MM-DD_HH-MM-SS
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

    # Crear la carpeta de logs si no existe
    mkdir -p logs

    # Nombre del archivo de log basado en el nombre del config y timestamp
    BASE_NAME=$(basename "$CONFIG" .yaml)
    LOG_FILE="logs${LOG_DIR}/${BASE_NAME}__${TIMESTAMP}.log"  
    echo "🚀 Lanzando experimento con configuración: $CONFIG_PATH"
    echo "📂 Log: $LOG_FILE"

    echo "🚀 Lanzando experimento con configuración: $CONFIG_PATH"
    
    # Lanzar el script en segundo plano con nohup
    nohup python -u $TRAIN_SCRIPT --config "$CONFIG_PATH" > "$LOG_FILE" 2>&1 &

    echo "🔄 Experimento con $CONFIG_PATH en ejecución. Revisa tail -f $LOG_FILE para ver el progreso."
done

echo "✅ Todos los experimentos han sido lanzados en segundo plano."
