import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, roc_auc_score, matthews_corrcoef, precision_score, recall_score
from dataset_builder import extract_promoter_features

# --- 1. DEEP LEARNING ARCHITECTURE (PyTorch) ---
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()
        self.conv = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=15)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=5)
        self.dropout1 = nn.Dropout(0.2)
        # Bidirectional LSTM: 128 input features, 32 hidden size -> 64 output size
        self.lstm = nn.LSTM(input_size=128, hidden_size=32, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.transpose(1, 2) # Swap dimensions for LSTM (batch, seq_len, features)
        x, _ = self.lstm(x)
        x = x[:, -1, :] # Take the output of the last time step
        x = self.dropout2(x)
        x = self.fc(x)
        return self.sigmoid(x).squeeze()

def get_onehot_tensor(seqs):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    tensors = []
    for seq in seqs:
        encoded = [mapping.get(base, [0.25,0.25,0.25,0.25]) for base in seq]
        tensors.append(np.array(encoded).T) # Shape: (4, seq_len)
    return torch.tensor(np.array(tensors), dtype=torch.float32)

# --- 2. PIPELINE FUNCTIONS ---
def evaluate_model(y_true, y_proba, model_name):
    auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_proba >= optimal_threshold).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    return {'Model': model_name, 'Precision': precision, 'Recall': recall, 'MCC': mcc, 'AUC': auc}

def train_pytorch_model(model_name, train_seqs, y_train_labels, test_tensor, y_test, epochs=25):
    print(f"Training {model_name}...")
    X_train_tensor = get_onehot_tensor(train_seqs)
    y_train_tensor = torch.tensor(y_train_labels, dtype=torch.float32)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = CNN_BiLSTM()
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs): 
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        y_proba = model(test_tensor).numpy()
    return evaluate_model(y_test, y_proba, model_name)

def run_experiment(args):
    def load_seqs(path):
        with open(path, 'r') as f:
            return [line.strip().upper() for line in f if len(line.strip()) == 81]

    train_pos = load_seqs(args.train_pos)
    train_neg_std = load_seqs(args.train_neg_std)
    train_neg_aei = load_seqs(args.train_neg_aei)
    test_pos = load_seqs(args.test_pos)
    test_neg = load_seqs(args.test_neg)

    y_test = np.array([1]*len(test_pos) + [0]*len(test_neg))
    test_seqs = test_pos + test_neg
    X_test_tab = pd.DataFrame([extract_promoter_features(s) for s in test_seqs]).fillna(0)
    X_test_tensor = get_onehot_tensor(test_seqs)

    results = []

    # --- AEI DATA PREPARATION ---
    y_train_aei = np.array([1]*len(train_pos) + [0]*len(train_neg_aei))
    X_train_aei_tab = pd.DataFrame([extract_promoter_features(s) for s in train_pos + train_neg_aei]).fillna(0)
    
    # --- MODEL 1: AEI-LightGBM ---
    print("Training AEI-LightGBM...")
    lgbm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1)
    lgbm.fit(X_train_aei_tab, y_train_aei)
    y_proba_lgbm = lgbm.predict_proba(X_test_tab)[:, 1]
    results.append(evaluate_model(y_test, y_proba_lgbm, "AEI-LightGBM"))

    # --- MODEL 2: AEI-CNN+BiLSTM (NEW!) ---
    res_cnn_aei = train_pytorch_model("AEI-CNN+BiLSTM", train_pos + train_neg_aei, y_train_aei, X_test_tensor, y_test)
    results.append(res_cnn_aei)

    # --- STANDARD DATA PREPARATION ---
    y_train_std = np.array([1]*len(train_pos) + [0]*len(train_neg_std))
    X_train_std_tab = pd.DataFrame([extract_promoter_features(s) for s in train_pos + train_neg_std]).fillna(0)
    X_test_tab = X_test_tab.reindex(columns=X_train_std_tab.columns, fill_value=0)

    # --- MODEL 3: XGBoost (Standard) ---
    print("Training XGBoost (Standard Negatives)...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric='logloss', random_state=42)
    xgb.fit(X_train_std_tab, y_train_std)
    y_proba_xgb = xgb.predict_proba(X_test_tab)[:, 1]
    results.append(evaluate_model(y_test, y_proba_xgb, "Standard-XGBoost"))

    # --- MODEL 4: SVC (Standard) ---
    print("Training SVC (Standard Negatives)...")
    base_svc = SVC(kernel='rbf', random_state=42)
    svc = CalibratedClassifierCV(base_svc, ensemble=False)
    svc.fit(X_train_std_tab, y_train_std)
    y_proba_svc = svc.predict_proba(X_test_tab)[:, 1]
    results.append(evaluate_model(y_test, y_proba_svc, "Standard-SVC (RBF)"))

    # --- MODEL 5: CNN+BiLSTM (Standard) ---
    res_cnn_std = train_pytorch_model("Standard-CNN+BiLSTM", train_pos + train_neg_std, y_train_std, X_test_tensor, y_test)
    results.append(res_cnn_std)

    # --- OUTPUT EXACT LATEX TABLE METRICS ---
    print("\n" + "="*80)
    print("ULTIMATE SOTA COMPARISON METRICS (DBTBS BENCHMARK)")
    print("="*80)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False, float_format="%.3f"))
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pos", required=True)
    parser.add_argument("--train-neg-std", required=True)
    parser.add_argument("--train-neg-aei", required=True)
    parser.add_argument("--test-pos", required=True)
    parser.add_argument("--test-neg", required=True)
    args = parser.parse_args()
    run_experiment(args)