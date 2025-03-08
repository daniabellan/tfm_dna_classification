# Synthetic Data Generation Pipeline

## Prerequisites
- Conda installed with the `tfm` environment configured
- Required tools installed in the `tfm` environment:
  - [Squigulator](https://github.com/hasindu2008/squigulator)
  - [slow5tools](https://github.com/hasindu2008/slow5tools)
  - [Dorado](https://github.com/nanoporetech/dorado)
  - Python with necessary dependencies

## Pipeline Workflow

### 1. Reference Genome Verification
The script expects a reference genome file (`.fa` or `.fna`) inside the `genome_reference` directory. If no valid reference is found, the script will terminate with an error.

### 2. Synthetic Signal Generation (Squigulator)
Squigulator is used to generate synthetic electrical signal data from the reference genome.

### 3. Format Conversion (slow5tools)
The generated signal in BLOW5 format is converted to FAST5 format using slow5tools.

### 4. Basecalling (Dorado)
The FAST5 files are processed using Dorado to generate basecalled sequencing reads in SAM format.

### 5. Data Preprocessing (Python Script)
A Python script processes the basecalled data and structures it into an H5 file for training and testing purposes.

## Directory Structure
After execution, the script creates and organizes files within the following directory structure:

```
data/
│── <species_name>/
    │── genome_reference/       # Contains the reference genome (.fa or .fna)
    │── blow5/                  # Squigulator output (BLOW5 format)
    │── fast5/                  # Converted FAST5 format data
    │── basecalling_output/      # Dorado basecalling results (SAM format)
    │── final_data/              # Processed H5 dataset for training/testing
```

## Usage
Run the script as follows:
```bash
./script.sh <species_name>
```
Ensure that the Conda environment (`tfm`) is activated and all dependencies are installed before execution.

## Troubleshooting
- If the script fails to find a genome reference file, verify the presence of `.fa` or `.fna` files inside `<species_name>/genome_reference/`
- Ensure Squigulator, slow5tools, and Dorado are installed in the `tfm` Conda environment.
- If Python preprocessing fails, check the logs for missing dependencies or incorrect input paths.

