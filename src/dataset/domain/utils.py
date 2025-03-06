from itertools import product


def generate_kmer_dict(kmers_size: int):
    """
    Generates a dictionary of K-mers with their unique indices.

    This function creates all possible K-mers of a given size using the nucleotide bases 
    'A', 'C', 'G', and 'T'. Each K-mer is assigned a unique index in the dictionary.

    Parameters:
    kmers_size (int): The length of the K-mers to be generated.

    Returns:
    dict: A dictionary where keys are K-mers (strings) and values are their unique indices (integers).
    """
    bases = ['A', 'C', 'G', 'T']
    kmer_list = [''.join(p) for p in product(bases, repeat=kmers_size)]
    kmer_dict = {kmer: idx for idx, kmer in enumerate(kmer_list)}
    return kmer_dict
