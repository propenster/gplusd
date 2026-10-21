import os
import subprocess
from Bio import SeqIO

def fetch_cnnpromoter_data():
    os.makedirs("data", exist_ok=True)
    repo_dir = "data/CNNPromoterData"
    
    # 1. Clone the official CNNPromoter benchmark repository
    if not os.path.exists(repo_dir):
        print("Cloning the official CNNPromoter benchmark dataset...")
        subprocess.run(["git", "clone", "https://github.com/solovictor/CNNPromoterData.git", repo_dir])
    
    # 2. Extract B. subtilis (Gram-positive) SigmaA promoters
    pos_fasta = os.path.join(repo_dir, "B._subtilis", "sigmaA", "positives.fasta")
    neg_fasta = os.path.join(repo_dir, "B._subtilis", "sigmaA", "negatives.fasta")
    
    if not os.path.exists(pos_fasta):
        print(f"Error: Could not find {pos_fasta}. The repo structure might have changed.")
        return

    # 3. Format as clean text files (81bp) for your LightGBM model
    def fasta_to_txt(fasta_path, txt_path):
        sequences = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq).upper()
            if len(seq) == 81 and 'N' not in seq:
                sequences.append(seq)
        
        with open(txt_path, "w") as f:
            for seq in sequences:
                f.write(f"{seq}\n")
        return len(sequences)

    pos_count = fasta_to_txt(pos_fasta, "data/dbtbs_positives.txt")
    neg_count = fasta_to_txt(neg_fasta, "data/dbtbs_negatives.txt")
    
    print(f"Successfully formatted {pos_count} TSS-aligned DBTBS positives.")
    print(f"Successfully formatted {neg_count} TSS-aligned DBTBS negatives.")
    print("Ready for external evaluation!")

if __name__ == "__main__":
    fetch_cnnpromoter_data()