import numpy as np
import pandas as pd
from generate_negative_sequences import generate_adversarial_negative

def extract_promoter_features(sequence, window_size=10):
    """
    Extracts tabular features that survive De Bruijn scrambling:
    1. Positional One-Hot 
    2. Local GC-Skew
    3. Thermodynamic Stability Proxy
    """
    sequence = sequence.upper()
    seq_len = len(sequence)
    features = {}
    
    # 1. Positional One-Hot Encoding
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    for i, base in enumerate(sequence):
        encoded = mapping.get(base, [0,0,0,0])
        features[f'pos_{i}_A'] = encoded[0]
        features[f'pos_{i}_C'] = encoded[1]
        features[f'pos_{i}_G'] = encoded[2]
        features[f'pos_{i}_T'] = encoded[3]
        
    # 2. Local GC-Skew & 3. Thermodynamic Stability Proxy
    stability_map = {'A': 2, 'T': 2, 'G': 3, 'C': 3}
    for i in range(seq_len - window_size + 1):
        window = sequence[i:i+window_size]
        
        # GC Skew
        g_count = window.count('G')
        c_count = window.count('C')
        skew = (g_count - c_count) / (g_count + c_count) if (g_count + c_count) > 0 else 0.0
        features[f'gc_skew_win_{i}'] = skew
        
        # Thermodynamic Stability (Hydrogen bonds)
        stability = sum(stability_map.get(base, 2.5) for base in window) / window_size
        features[f'stability_win_{i}'] = stability
        
    return features

def prepare_dataset(data_path, k_size=3):
    """Loads positive sequences, generates AEI negatives, and builds feature table."""
    print(f"Loading positive sequences from {data_path}...")
    with open(data_path, 'r') as f:
        positives = [line.strip() for line in f if line.strip()]

    print(f"Generating AEI negatives and extracting features...")
    data_rows = []
    
    for pos_seq in positives:
        # Positive Sample
        pos_features = extract_promoter_features(pos_seq)
        pos_features['label'] = 1
        data_rows.append(pos_features)
        
        # Negative Sample (De Bruijn Adversarial)
        neg_seq = generate_adversarial_negative(pos_seq, k=k_size)
        if not neg_seq: # Fallback if graph fails
            print("Fall back because graph fails...")
            neg_seq = pos_seq
        neg_features = extract_promoter_features(neg_seq)
        neg_features['label'] = 0
        data_rows.append(neg_features)
        
    return pd.DataFrame(data_rows)