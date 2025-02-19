#!/bin/bash

# Verifica si se proporciona la especie como argumento
if [ "$#" -ne 1 ]; then
    echo "Uso: $0 <nombre_especie>"
    echo "Debe haber un archivo de referencia en: <nombre_especie>/genome_reference/ con extensión .fa o .fna"
    exit 1
fi

SPECIE=$1
BASE_DIR="data/${SPECIE}"
GENOME_DIR="${BASE_DIR}/genome_reference"
CONDA_ENV="tfm"

# Buscar el archivo de referencia en genome_reference (extensiones .fa o .fna)
GENOME_REF=$(find "$GENOME_DIR" -maxdepth 1 -type f \( -name "*.fa" -o -name "*.fna" \) | head -n 1)

# Verificar si se encontró un archivo válido
if [ -z "$GENOME_REF" ]; then
    echo "Error: No se encontró ningún archivo de referencia en $GENOME_DIR con extensión .fa o .fna"
    exit 1
fi

echo "=== Usando genoma de referencia: $GENOME_REF ==="

# Crear las carpetas necesarias
mkdir -p "${BASE_DIR}/blow5" # Datos generados artificialmente con Squigulator
mkdir -p "${BASE_DIR}/fast5" # Datos transformados de blow5 a fast5
mkdir -p "${BASE_DIR}/basecalling_output" # Salida del basecalling de Dorado
mkdir -p "${BASE_DIR}/final_data" # Archivo H5 para entrenamiento/test

# Activar el entorno Conda tfm
echo "=== Activando entorno Conda '$CONDA_ENV' ==="
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
if [ $? -ne 0 ]; then
    echo "Error: No se pudo activar el entorno Conda 'tfm'"
    exit 1
fi

echo "=== Generando señal eléctrica con Squigulator ==="
squigulator "$GENOME_REF" -x dna-r9-min -o "${BASE_DIR}/blow5/reads.blow5" -n 12000 --seed 42
if [ $? -ne 0 ]; then
    echo "Error al ejecutar Squigulator"
    exit 1
fi

echo "=== Convirtiendo BLOW5 a FAST5 con slow5tools ==="
slow5tools s2f "${BASE_DIR}/blow5/" -d "${BASE_DIR}/fast5/"
if [ $? -ne 0 ]; then
    echo "Error al ejecutar slow5tools"
    exit 1
fi

echo "=== Realizando basecalling con Dorado ==="
dorado basecaller --emit-sam -b 768 dorado_models/dna_r9.4.1_e8_sup@v3.6 "${BASE_DIR}/fast5/" -v > "${BASE_DIR}/basecalling_output/basecalled_signal.sam"
if [ $? -ne 0 ]; then
    echo "Error al ejecutar Dorado"
    exit 1
fi

echo "=== Aplicando preprocesamiento con Python en entorno Conda (tfm) ==="
# Ejecutar el script de Python con los argumentos correctos
python datasets/create_dataset_from_pod5_fast5.py $BASE_DIR/basecalling_output/basecalled_signal.sam $BASE_DIR/fast5 $BASE_DIR/final_data/matched_data.h5 --verbose
if [ $? -ne 0 ]; then
    echo "Error al ejecutar el preprocesamiento con Python"
    exit 1
fi

echo "=== Proceso completado con éxito ==="
