import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shap
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



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

def get_onehot_tensor(seqs):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    tensors = [np.array([mapping.get(b, [0.25]*4) for b in s]).T for s in seqs]
    return torch.tensor(np.array(tensors), dtype=torch.float32)


def main(pos_path, neg_path, model_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model = CNN_BiLSTM()
    model.load_state_dict(torch.load(Path(model_dir) / "best_model.pt"))
    model.eval()
    
    with open(pos_path, 'r') as f:
        pos_seqs = [line.strip() for line in f if len(line.strip()) == 81]
    with open(neg_path, 'r') as f:
        neg_seqs = [line.strip() for line in f if len(line.strip()) == 81]
        
    X_tensor = get_onehot_tensor(pos_seqs + neg_seqs)
    
    print("Calculating SHAP GradientExplainer values (LSTM compatible)...")
    # Use a random background of 100 sequences to initialize GradientExplainer
    background = X_tensor[np.random.choice(X_tensor.shape[0], 100, replace=False)]
    
    # 1. Switch to GradientExplainer
    explainer = shap.GradientExplainer(model, background)
    
    # Explain the positive sequences
    test_tensor = get_onehot_tensor(pos_seqs[:100])
    shap_values = explainer.shap_values(test_tensor)
    
    # 2. Safely extract the numpy array
    if isinstance(shap_values, (list, tuple)):
        shap_values = shap_values[0]
    if hasattr(shap_values, 'detach'):
        shap_values = shap_values.detach().cpu().numpy()
        
    # Force the array into the exact shape we expect: (Samples, 4 nucleotides, 81 base pairs)
    shap_values = np.array(shap_values).reshape(-1, 4, 81)
        
    # Sum SHAP values across the 4 nucleotide channels (axis=1), then average across samples (axis=0)
    positional_shap = np.abs(shap_values).sum(axis=1).mean(axis=0)
    
    # Guarantee it is a flat 1D array of 81 values for matplotlib
    positional_shap = positional_shap.flatten()
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(81), positional_shap, color='darkblue')
    plt.title("DeepExplainer Positional Importance (AEI-CNN+BiLSTM)", fontweight='bold')
    plt.xlabel("Position in Sequence (bp)", fontweight='bold')
    plt.ylabel("Mean |SHAP Value|", fontweight='bold')
    plt.xticks([0, 20, 40, 60, 80], ['-60', '-40', '-20', '+1', '+20'])
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'shap_summary_plot.pdf', dpi=400)
    print("Saved PyTorch SHAP plot.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", default="data/dbtbs_positives.txt")
    parser.add_argument("--neg", default="data/dbtbs_negatives.txt")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", default="figs")
    args = parser.parse_args()
    main(args.pos, args.neg, args.model_dir, args.out_dir)