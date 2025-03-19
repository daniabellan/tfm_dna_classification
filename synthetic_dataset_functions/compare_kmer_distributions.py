import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import ks_2samp, chisquare
from scipy.spatial.distance import jensenshannon


DATA_ROOT_PATH = "data"

for SPECIE in os.listdir(DATA_ROOT_PATH):
    # Avoid real data
    if SPECIE == "ecoli_k12_real": continue
    
    # Define file paths based on species name
    BASE_DIR = f"{DATA_ROOT_PATH}/{SPECIE}/jellyfish_output"
    DORADO_FILE_PATH = f"{BASE_DIR}/{SPECIE}_genome.txt"  # K-mers from Dorado (Basecalling)
    GENOME_FILE_PATH = f"{BASE_DIR}/{SPECIE}_bam.txt"  # K-mers from the reference genome

    if SPECIE == "mm39":
        specie_name = "Mus Musculus"
    elif SPECIE == "ecoli_O104H4":
        specie_name = "E. Coli O104 H4"
    elif SPECIE == "ecoli_O157H7":
        specie_name = "E. Coli O157 H7"
    elif SPECIE == "salmonella_enterica":
        specie_name = "Salmonella Enterica"
    else:
        specie_name = SPECIE

    # Load data from files
    dorado_df = pd.read_csv(DORADO_FILE_PATH, sep="\t", names=["K-mer", "Frequency"], on_bad_lines="skip")
    genome_df = pd.read_csv(GENOME_FILE_PATH, sep="\t", names=["K-mer", "Frequency"], on_bad_lines="skip")

    # Normalize frequencies
    dorado_df["Normalized_Freq"] = dorado_df["Frequency"] / dorado_df["Frequency"].sum()
    genome_df["Normalized_Freq"] = genome_df["Frequency"] / genome_df["Frequency"].sum()

    # Merge datasets on K-mers
    merged_df = pd.merge(dorado_df, genome_df, on="K-mer", suffixes=("_dorado", "_genome"))

    # Perform statistical comparisons
    ks_stat, ks_p = ks_2samp(merged_df["Normalized_Freq_dorado"], merged_df["Normalized_Freq_genome"])
    js_div = jensenshannon(merged_df["Normalized_Freq_dorado"], merged_df["Normalized_Freq_genome"])
    chi_stat, chi_p = chisquare(merged_df["Normalized_Freq_dorado"], merged_df["Normalized_Freq_genome"])

    statistical_values = {
        "KS Test Statistic": f"{ks_stat:.4f}",
        "KS Test p-value": f"{ks_p:.4f}",
        "Jensen-Shannon Distance": f"{js_div:.4f}",
        "Chi-Square Statistic": f"{chi_stat:.4f}",
        "Chi-Square p-value": f"{chi_p:.4f}"
    }

    # Output statistical results with explanations
    print(f"\nStatistical Results and Interpretation ({SPECIE}):")
    for key, explanation in statistical_values.items():
        print(f"{key}: {explanation}")

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.bar(merged_df["K-mer"], merged_df["Normalized_Freq_dorado"], alpha=0.5, label="Dorado (Basecalling)", color="red")
    plt.bar(merged_df["K-mer"], merged_df["Normalized_Freq_genome"], alpha=0.5, label="Reference Genome", color="blue")
    plt.xlabel("K-mers")
    plt.ylabel("Normalized Frequency")
    plt.xticks(rotation=90)
    plt.legend()
    plt.title(f"K-mer Distribution Comparison ({specie_name})")
    plt.show()

pass