import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shap
import torch
import torch.nn as nn

# 1. Define the model blueprint
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

# 2. Define the tensor encoder
def get_onehot_tensor(seqs):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    tensors = [np.array([mapping.get(b, [0.25]*4) for b in s]).T for s in seqs]
    return torch.tensor(np.array(tensors), dtype=torch.float32)

def main(pos_path, neg_path, model_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading PyTorch model from {model_dir}...")
    model = CNN_BiLSTM()
    model.load_state_dict(torch.load(Path(model_dir) / "best_model.pt", map_location=torch.device('cpu')))
    model.eval()
    
    print("Loading external sequences...")
    with open(pos_path, 'r') as f:
        pos_seqs = [line.strip() for line in f if len(line.strip()) == 81]
    with open(neg_path, 'r') as f:
        neg_seqs = [line.strip() for line in f if len(line.strip()) == 81]
        
    print("Converting sequences to PyTorch tensors...")
    # SHAP is computationally heavy for Deep Learning. 
    # We use 100 negatives as a baseline background, and evaluate 200 positive predictions.
    bg_tensor = get_onehot_tensor(neg_seqs[:100])
    test_tensor = get_onehot_tensor(pos_seqs[:200])
    
    print("Calculating SHAP values via GradientExplainer (this might take a minute)...")
    explainer = shap.GradientExplainer(model, bg_tensor)
    shap_values = explainer.shap_values(test_tensor)
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values
        
    # SHAP outputs shape (Samples, 4 channels, 81 bp). 
    # We sum the absolute SHAP values across the 4 nucleotide channels to get the importance of the POSITION itself.
    positional_shap = np.sum(np.abs(shap_vals), axis=1)
    
    # Create labels mapping to the 0-80 indices
    feature_names = [f"Position {i}" for i in range(81)]
    df_features = pd.DataFrame(np.zeros((len(test_tensor), 81)), columns=feature_names)

    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(12, 8))
    
    shap.summary_plot(
        positional_shap, 
        df_features, 
        max_display=20,  # Show top 20 most important positional coordinates
        show=False,
        plot_size=(12, 8)
    )
    
    plt.title("SHAP Positional Importance: AEI-CNN+BiLSTM", fontweight='bold', fontsize=16, pad=20)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'shap_summary_plot.pdf'
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"Saved SHAP plot to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", type=str, default="data/cnnprom_positive.txt", help="External positives")
    parser.add_argument("--neg", type=str, default="data/cnnprom_negative.txt", help="External negatives")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with best_model.pt")
    parser.add_argument("--out-dir", type=str, default="figs", help="Output directory for plots")

    args = parser.parse_args()
    main(args.pos, args.neg, args.model_dir, args.out_dir)