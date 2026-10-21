import argparse
import os
import random
from Bio import Entrez, SeqIO
from pathlib import Path

# Entrez requires an email to know who is querying their servers
Entrez.email = "faith508@iastate.edu" 

def generate_external_dataset(pos_out="data/cnnprom_positive.txt", neg_out="data/cnnprom_negative.txt", seq_length=81):
    os.makedirs("data", exist_ok=True)
    
    print("Fetching Bacillus subtilis reference genome (NC_000964.3) directly from NCBI...")
    try:
        # Fetch the full annotated GenBank file
        handle = Entrez.efetch(db="nucleotide", id="NC_000964.3", rettype="gbwithparts", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
    except Exception as e:
        print(f"Failed to fetch from NCBI. Error: {e}")
        return

    genome_seq = record.seq
    positives = []
    
    print(f"Length of genome fetched from NCBI >>> {len(genome_seq)}")
    print("Extracting upstream regulatory regions (-60 to +20) for all CDS...")
    
    # Extract upstream regions of annotated genes as presumed true promoters
    for feature in record.features:
        if feature.type == "CDS":
            # FIXED: Access strand through feature.location
            if feature.location.strand == 1: # Forward strand
                tss = feature.location.start
                start = tss - 60
                end = tss + 21
                if start >= 0:
                    seq = str(genome_seq[start:end]).upper()
                    if len(seq) == seq_length and 'N' not in seq:
                        positives.append(seq)
            elif feature.location.strand == -1: # Reverse strand
                tss = feature.location.end
                start = tss - 21
                end = tss + 60
                if end <= len(genome_seq):
                    # Remember to reverse complement the reverse strand!
                    seq = str(genome_seq[start:end].reverse_complement()).upper()
                    if len(seq) == seq_length and 'N' not in seq:
                        positives.append(seq)
                        
    # Save Positives
    with open(pos_out, "w") as f:
        for seq in positives:
            f.write(f"{seq}\n")
    print(f"Saved {len(positives)} putative promoters to {pos_out}")

    # Generate Negatives
    print("Generating background genomic negatives...")
    negatives = []
    genome_str = str(genome_seq).upper()
    
    while len(negatives) < len(positives):
        start = random.randint(0, len(genome_str) - seq_length)
        chunk = genome_str[start:start+seq_length]
        if 'N' not in chunk:
            negatives.append(chunk)
            
    with open(neg_out, "w") as f:
        for seq in negatives:
            f.write(f"{seq}\n")
    print(f"Saved {len(negatives)} real genome background sequences to {neg_out}")
    print("External dataset preparation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare external benchmark dataset for CNNProm.")
    parser.add_argument("--pos-out", type=str, default="data/cnnprom_positive.txt", help="Output file for positive sequences")
    parser.add_argument("--neg-out", type=str, default="data/cnnprom_negative.txt", help="Output file for negative sequences")
    parser.add_argument("--seq-length", type=int, default=81, help="Length of sequences to extract")
    args = parser.parse_args()

    generate_external_dataset(pos_out=args.pos_out, neg_out=args.neg_out, seq_length=args.seq_length)
