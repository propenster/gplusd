import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import logomaker
import os
from dataset_builder import extract_promoter_features

def plot_roc_pr_curves(y_true, y_proba, output_dir):
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AEI-LGBM (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.set_title('ROC Curve (External Benchmark)', fontweight='bold')
    ax1.legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    ax2.plot(recall, precision, color='green', lw=2, label=f'AEI-LGBM (AP = {ap:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision-Recall Curve (External Benchmark)', fontweight='bold')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'roc_pr_curves.png', dpi=400)
    print("Saved ROC and PR curves.")

def generate_comparative_logo(pos_file, neg_file, output_dir):
    """Generates the stacked Positives vs AEI Negatives conservation plot."""
    with open(pos_file, 'r') as f:
        pos_seqs = [line.strip().upper() for line in f if line.strip()]
    with open(neg_file, 'r') as f:
        neg_seqs = [line.strip().upper() for line in f if line.strip()]

    def get_info_matrix(seqs):
        seq_len = len(seqs[0])
        counts = pd.DataFrame(0, index=range(seq_len), columns=['A', 'C', 'G', 'T'])
        for seq in seqs:
            for i, base in enumerate(seq[:seq_len]):
                if base in counts.columns:
                    counts.loc[i, base] += 1
        prob_matrix = counts.div(counts.sum(axis=1), axis=0)
        return logomaker.transform_matrix(prob_matrix, from_type='probability', to_type='information')

    pos_info = get_info_matrix(pos_seqs)
    neg_info = get_info_matrix(neg_seqs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, sharey=True)

    logomaker.Logo(pos_info, ax=ax1, color_scheme='classic')
    ax1.set_title('Universal Prokaryotic Promoters (Positive Set)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('bits', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 2.0])

    logomaker.Logo(neg_info, ax=ax2, color_scheme='classic')
    ax2.set_title('De Bruijn AEI Non-Promoters (Negative Set)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('bits', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sequence Position', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'comparative_sequence_logo.png', dpi=400)
    print("Saved Comparative Sequence Logo.")

def main(ext_pos, ext_neg, train_pos, train_neg, model_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(Path(model_dir) / "best_model.pkl")
    feature_cols = joblib.load(Path(model_dir) / "feature_columns.pkl")
    
    with open(ext_pos, 'r') as f:
        pos_seqs = [line.strip() for line in f if line.strip()]
    with open(ext_neg, 'r') as f:
        neg_seqs = [line.strip() for line in f if line.strip()]
        
    all_seqs = pos_seqs + neg_seqs
    y_true = np.array([1]*len(pos_seqs) + [0]*len(neg_seqs))
    
    features_list = [extract_promoter_features(seq) for seq in all_seqs]
    df = pd.DataFrame(features_list).reindex(columns=feature_cols, fill_value=0)
    
    y_proba = model.predict(df)
    
    plot_roc_pr_curves(y_true, y_proba, output_dir)
    
    # Generate the comparative logo using the training data files
    if train_pos and train_neg:
        generate_comparative_logo(train_pos, train_neg, output_dir)
        
    print(f"Successfully generated paper figures in '{output_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # External test sets for ROC/PR curves
    parser.add_argument("--ext-pos", type=str, default="data/dbtbs_positives.txt")
    parser.add_argument("--ext-neg", type=str, default="data/dbtbs_negatives.txt")
    
    # Training sets for Comparative Logo
    parser.add_argument("--train-pos", type=str, help="e.g., data/gplus.txt")
    parser.add_argument("--train-neg", type=str, help="e.g., data/gplus_negative.txt")
    
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="figs2")

    args = parser.parse_args()

    main(args.ext_pos, args.ext_neg, args.train_pos, args.train_neg, args.model_dir, args.out_dir)