Commands:
pod5 convert fast5 data/ecoli_k12/fast5_1 -o data/ecoli_k12/pod5_1 --one-to-one data/ecoli_k12/fast5_1

dorado basecaller --emit-sam sup@latest data/ecoli_k12/pod5_1 -v > data/ecoli_k12/output.sam

dorado basecaller --emit-sam -b 768 data/dorado_models/dna_r9.4.1_e8_sup@v3.6 outputs/ -v > outputs/output.sam