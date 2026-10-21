import argparse
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from sklearn.metrics import roc_curve, roc_auc_score, recall_score
import torch.nn as nn
import torch


class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()
        self.conv = nn.Conv1d(4, 128, 15)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(5)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(128, 32, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.pool(self.relu(self.conv(x))))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return self.sigmoid(self.fc(self.dropout2(x[:, -1, :])))

def parse_fasta_sigmas(fasta_path):
    """Parse FASTA headers formatted like '>abnA_SigA_2950133 DBTBS_benchmark_81bp'."""
    records = list(SeqIO.parse(fasta_path, "fasta"))
    data = []
    
    for rec in records:
        # rec.id gets sequence identifier prior to spaces (e.g., abnA_SigA_2950133)
        header_id = rec.id.split()[0]
        parts = header_id.split('_')
        
        # Extract sigma factor (e.g., 'SigA', 'SigB', 'SigK')
        sigma = "Unknown"
        for part in parts:
            if re.match(r'^Sig[A-Z0-9]+$', part, re.IGNORECASE):
                sigma = part.capitalize()
                break
                
        data.append({
            "id": header_id,
            "sigma": sigma,
            "sequence": str(rec.seq).upper()
        })
        
    df = pd.DataFrame(data)
    df.to_csv("dbtbs_benchmark_parsed.csv", index=False)
    print(f"Parsed {len(df)} sequences across {df['sigma'].nunique()} sigma factors:")
    print(df['sigma'].value_counts().to_string())
    return df

def one_hot_encode(sequences, channel_first=True):
    """Convert DNA sequences to one-hot tensors for CNN inputs."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded_list = []
    
    for seq in sequences:
        arr = np.zeros((4, len(seq)), dtype=np.float32)
        for i, base in enumerate(seq):
            if base in mapping:
                arr[mapping[base], i] = 1.0
        if not channel_first:
            arr = arr.T  # (L, 4)
        encoded_list.append(arr)
        
    return np.array(encoded_list)


def get_onehot_tensor(seqs):
    """The exact tensor encoder used during GPlusD training."""
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    tensors = [np.array([mapping.get(b, [0.25]*4) for b in s]).T for s in seqs]
    return torch.tensor(np.array(tensors), dtype=torch.float32)

class ModelPredictor:
    """Unified predictor wrapper handling CNNs (PyTorch/TF) or feature-based models."""
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model_type = None
        self.model = None
        self.feature_cols = None
        self._load_model()

    def _load_model(self):
        pt_path = self.model_dir / "best_model.pt"
        # Create the model structure first
        self.model = CNN_BiLSTM()
        # Pour the .pt weights into the structure
        self.model.load_state_dict(torch.load(pt_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.model_type = "pytorch_cnn"
        print(f"Loaded PyTorch CNN model from {pt_path}")
        return

    # def predict_proba(self, sequences):
    #     X = get_onehot_tensor(sequences)
    #     with torch.no_grad():
    #         out = self.model(X)
    #         if out.shape[-1] == 1 or len(out.shape) == 1:
    #             probs = torch.sigmoid(out).squeeze().numpy()
    #         else:
    #             probs = torch.softmax(out, dim=1)[:, 1].numpy()
    #     return np.atleast_1d(probs)
    def predict_proba(self, sequences):
        X = get_onehot_tensor(sequences)
        with torch.no_grad():
            out = self.model(X)
            # FIX: The model already applies sigmoid natively. Do not apply it twice.
            probs = out.squeeze().numpy()
        return np.atleast_1d(probs)

def main(fasta_path, neg_path, model_dir, output_dir, min_samples=10):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load Model Predictor
    predictor = ModelPredictor(model_dir)
    
    # Parse Benchmark Data
    pos_df = parse_fasta_sigmas(fasta_path)
    
    # Load Background Negatives
    with open(neg_path, 'r') as f:
        neg_seqs = [line.strip().upper() for line in f if len(line.strip()) == 81]
    
    print(f"Loaded {len(neg_seqs)} background negative sequences.")
    neg_probs = predictor.predict_proba(neg_seqs)
    
    # Group by Sigma Factor and Compute Metrics
    sigmas = pos_df['sigma'].value_counts()
    valid_sigmas = sigmas[sigmas >= min_samples].index.tolist()
    
    fig, (ax_roc, ax_bar) = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_theme(style="whitegrid")
    
    metrics = []
    
    for sigma in valid_sigmas:
        sig_seqs = pos_df[pos_df['sigma'] == sigma]['sequence'].tolist()
        pos_probs = predictor.predict_proba(sig_seqs)
        
        # Combine positives for this sigma with background negatives
        y_true = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
        y_score = np.concatenate([pos_probs, neg_probs])
        
        # Compute Metrics
        auc = roc_auc_score(y_true, y_score)
        sensitivity = np.mean(pos_probs >= 0.5)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        
        metrics.append({
            "Sigma Factor": sigma,
            "Count": len(sig_seqs),
            "AUC": auc,
            "Sensitivity": sensitivity
        })
        
        # Plot ROC curve for each sigma factor
        ax_roc.plot(fpr, tpr, label=f"{sigma} (n={len(sig_seqs)}, AUC={auc:.3f})", lw=2)

    # Styling ROC Panel
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
    ax_roc.set_title('ROC Curves Stratified by Sigma Factor', fontweight='bold', fontsize=14)
    ax_roc.legend(loc="lower right", frameon=True)

    # Plot Performance Metrics Bar Chart
    metrics_df = pd.DataFrame(metrics)
    
    # CORRECTED: Use 'value_name' instead of 'value_value_name'
    melted_df = pd.melt(metrics_df, id_vars=['Sigma Factor', 'Count'], 
                        value_vars=['AUC', 'Sensitivity'], 
                        var_name='Metric', value_name='Score')
    
    melted_df['Sigma Label'] = melted_df.apply(lambda r: f"{r['Sigma Factor']}\n(n={r['Count']})", axis=1)
    
    # CORRECTED: y axis must match 'Score'
    sns.barplot(data=melted_df, x='Sigma Label', y='Score', hue='Metric', 
                palette=['#1f77b4', '#2ca02c'], ax=ax_bar)
    
    ax_bar.set_ylim([0.0, 1.05])
    ax_bar.set_ylabel('Score', fontweight='bold')
    ax_bar.set_xlabel('Sigma Factor', fontweight='bold')
    ax_bar.set_title('Model Performance Metrics', fontweight='bold', fontsize=14)
    ax_bar.legend(loc="lower right")

    plt.tight_layout()
    plot_file = Path(output_dir) / 'sigma_factor_performance.pdf'
    plt.savefig(plot_file, dpi=600)
    print(f"\nSaved performance evaluation figure to {plot_file}")
    
    # Output Table
    print("\nDetailed Performance Table:")
    print(metrics_df.to_string(index=False, float_format="%.4f"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model across sigma factors.")
    parser.add_argument("--fasta", type=str, default="dbtbs_benchmark.fasta", help="FASTA file containing benchmark sequences")
    parser.add_argument("--neg", type=str, default="data/dbtbs_negatives.txt", help="Background negative sequence file")
    parser.add_argument("--model-dir", type=str, default="models/aei_cnn", help="Directory containing the model checkpoint")
    parser.add_argument("--out-dir", type=str, default="figs", help="Output directory for plots")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum sequences required to evaluate a sigma factor")
    
    args = parser.parse_args()
    main(args.fasta, args.neg, args.model_dir, args.out_dir, args.min_samples)