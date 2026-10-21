import argparse
import os
from generate_negative_sequences import generate_adversarial_negative

def main(input_file, output_file, k=3):
    print(f"Reading positive sequences from {input_file}...")
    with open(input_file, 'r') as f:
        positives = [line.strip().upper() for line in f if line.strip()]
        
    print(f"Generating AEI negative sequences (k={k})... This may take a minute.")
    negatives = []
    for seq in positives:
        neg_seq = generate_adversarial_negative(seq, k=k)
        # Fallback if the graph couldn't find a divergent Eulerian path
        if not neg_seq:
            neg_seq = seq 
        negatives.append(neg_seq)
        
    with open(output_file, 'w') as f:
        for seq in negatives:
            f.write(f"{seq}\n")
            
    print(f"Successfully saved {len(negatives)} AEI negative sequences to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to positive sequences (e.g., data/prokaryote_promoters.txt)")
    parser.add_argument("--output", type=str, required=True, help="Path to save negative sequences")
    parser.add_argument("-k", type=int, default=3, help="K-mer size for De Bruijn graph")
    args = parser.parse_args()
    main(args.input, args.output, args.k)