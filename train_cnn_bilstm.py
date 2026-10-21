import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from pathlib import Path

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
        return self.sigmoid(self.fc(self.dropout2(x[:, -1, :]))).squeeze()

def get_onehot_tensor(seqs):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    tensors = [np.array([mapping.get(b, [0.25]*4) for b in s]).T for s in seqs]
    return torch.tensor(np.array(tensors), dtype=torch.float32)

def train_pytorch(data_path, neg_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(data_path, 'r') as f:
        pos = [l.strip().upper() for l in f if len(l.strip()) == 81]
    with open(neg_path, 'r') as f:
        neg = [l.strip().upper() for l in f if len(l.strip()) == 81]
        
    X_tensor = get_onehot_tensor(pos + neg)
    y = np.array([1]*len(pos) + [0]*len(neg))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_mcc = -1
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        train_loader = DataLoader(TensorDataset(X_tensor[train_idx], torch.tensor(y[train_idx], dtype=torch.float32)), batch_size=64, shuffle=True)
        model = CNN_BiLSTM()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        
        model.train()
        for epoch in range(20):
            for bx, by in train_loader:
                optimizer.zero_grad()
                criterion(model(bx), by).backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            preds = model(X_tensor[val_idx]).numpy()
        
        mcc = matthews_corrcoef(y[val_idx], (preds >= 0.5).astype(int))
        print(f"Fold {fold}: MCC = {mcc:.4f} | AUC = {roc_auc_score(y[val_idx], preds):.4f}")
        
        if mcc > best_mcc:
            best_mcc = mcc
            torch.save(model.state_dict(), Path(output_dir) / "best_model.pt")
            print(f" -> Saved new best model (MCC: {mcc:.4f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", required=True)
    parser.add_argument("--neg", required=True)
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()
    train_pytorch(args.pos, args.neg, args.out_dir)