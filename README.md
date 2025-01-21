Commands:
pod5 convert fast5 data/ecoli_k12/fast5_1 -o data/ecoli_k12/pod5_1 --one-to-one data/ecoli_k12/fast5_1

dorado basecaller --emit-sam sup@latest data/ecoli_k12/pod5_1 -v > data/ecoli_k12/output.sam