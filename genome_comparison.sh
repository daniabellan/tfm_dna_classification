#!/bin/bash

# Activar Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Verificar que el entorno se activó correctamente
if [[ $? -ne 0 ]]; then
    echo "ERROR: No se pudo activar el entorno de Conda '$CONDA_ENV'."
    exit 1
fi

# Definir la carpeta base donde están las especies
BASE_DIR="data"

# Crear una lista de todos los archivos .fa y .fna dentro de los subdirectorios
GENOMES=($(find "$BASE_DIR" -type f \( -name "*.fa" -o -name "*.fna" \)))

# Carpeta de salida para los resultados
OUTPUT_DIR="fastani_results"
mkdir -p "$OUTPUT_DIR"

# Archivo de salida CSV
OUTPUT_FILE="$OUTPUT_DIR/ANI_results.csv"
echo "Query,Reference,ANI" > "$OUTPUT_FILE"

# Archivo de log de errores
ERROR_LOG="$OUTPUT_DIR/error_log.txt"
> "$ERROR_LOG"  # Limpia el archivo antes de empezar

# Loop para comparar todas las especies con todas
for query in "${GENOMES[@]}"; do
    for reference in "${GENOMES[@]}"; do
        if [[ "$query" != "$reference" ]]; then
            echo "🔍 Comparando: $query vs $reference"

            # Ejecutar FastANI y capturar la salida
            fastANI -q "$query" -r "$reference" --output "$OUTPUT_DIR/temp_output.txt" 2>> "$ERROR_LOG"

            # Si la ejecución fue exitosa, extraer el ANI
            if [[ -s "$OUTPUT_DIR/temp_output.txt" ]]; then
                ANI_VALUE=$(awk '{print $3}' "$OUTPUT_DIR/temp_output.txt")
                echo "$query,$reference,$ANI_VALUE" >> "$OUTPUT_FILE"
            else
                echo "$query,$reference,ERROR" >> "$OUTPUT_FILE"
            fi
        fi
    done
done

# Limpiar archivo temporal
rm -f "$OUTPUT_DIR/temp_output.txt"

echo "✅ Comparación terminada. Resultados guardados en $OUTPUT_FILE"
echo "⚠️  Si hubo errores, revisa $ERROR_LOG"
