import os
import random
from Bio import SeqIO

def fasta_to_txt(fasta_path, txt_path, max_samples=None):
    sequences = []
    # Make sure the data folder exists
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        # Keep sequences that are exactly 81bp and have no unknown 'N' bases
        if len(seq) == 81 and 'N' not in seq:
            sequences.append(seq)
            
    # If a max limit is set, randomly downsample to balance the dataset
    if max_samples is not None and len(sequences) > max_samples:
        sequences = random.sample(sequences, max_samples)
    
    with open(txt_path, "w") as f:
        for seq in sequences:
            f.write(f"{seq}\n")
    return len(sequences)

if __name__ == "__main__":
    random.seed(42) # Set seed for reproducibility
    
    # 1. Process Positives first to get the target count
    pos_count = fasta_to_txt("data/CNNPromoterData/Bacillus_prom.fa", "data/dbtbs_positives.txt")
    
    # 2. Process Negatives and cap them at the positive count
    neg_count = fasta_to_txt("data/CNNPromoterData/Bacillus_non_prom.fa", "data/dbtbs_negatives.txt", max_samples=pos_count)
    
    print(f"Successfully formatted {pos_count} TSS-aligned DBTBS positives.")
    print(f"Successfully formatted {neg_count} TSS-aligned DBTBS negatives (balanced).")