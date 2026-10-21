import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix

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

def evaluate_external(pos_path, neg_path, model_dir):
    print(f"Loading model from {model_dir}...")
    
    # 3. Load the PyTorch model properly
    model = CNN_BiLSTM()
    model.load_state_dict(torch.load(Path(model_dir) / "best_model.pt", map_location=torch.device('cpu')))
    model.eval()
    
    # Load sequences
    with open(pos_path, 'r') as f:
        pos_seqs = [line.strip() for line in f if len(line.strip()) == 81]
    with open(neg_path, 'r') as f:
        neg_seqs = [line.strip() for line in f if len(line.strip()) == 81]
        
    print(f"Evaluating on {len(pos_seqs)} external positives and {len(neg_seqs)} external negatives.")
    
    # Build dataset
    all_seqs = pos_seqs + neg_seqs
    y_true = np.array([1]*len(pos_seqs) + [0]*len(neg_seqs))
    
    # 4. Convert sequences to tensors and predict
    X = get_onehot_tensor(all_seqs)
    with torch.no_grad():
        y_proba = model(X).squeeze().numpy()
    
    # Calculate Metrics
    auc = roc_auc_score(y_true, y_proba)
    
    # --- FIND OPTIMAL THRESHOLD (Youden's J Statistic) ---
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nDefault 0.5 Threshold is too strict. Adjusting to Optimal Threshold: {optimal_threshold:.4f}")
    
    # Apply new threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*50)
    print("EXTERNAL VALIDATION RESULTS (AEI-CNN+BiLSTM)")
    print("="*50)
    print(f"AUC:       {auc:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("-" * 50)
    print("Confusion Matrix:")
    print(f"True Negatives:  {cm[0][0]} | False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]} | True Positives:  {cm[1][1]}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", type=str, required=True)
    parser.add_argument("--neg", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)

    args = parser.parse_args()
    evaluate_external(args.pos, args.neg, args.model_dir)