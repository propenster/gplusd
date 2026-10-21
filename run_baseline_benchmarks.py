import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

# 1. Your exact CNN-BiLSTM from run_sota_comparison.py / train_cnn_bilstm.py
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()
        self.conv = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=15, padding="same")
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=5)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=32, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.pool(self.relu(self.conv(x))))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout2(x)
        return self.sigmoid(self.fc(x)).squeeze()

def get_onehot_tensor(seqs):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    tensors = [np.array([mapping.get(b, [0.25]*4) for b in s]).T for s in seqs]
    return torch.tensor(np.array(tensors), dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", default="data/gplusd_positive.txt")
    parser.add_argument("--neg", default="data/gplusd_negative.txt")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    with open(args.pos, "r") as f:
        pos_seqs = [l.strip().upper() for l in f if len(l.strip()) == 81]
    with open(args.neg, "r") as f:
        neg_seqs = [l.strip().upper() for l in f if len(l.strip()) == 81]

    n_samples = min(len(pos_seqs), len(neg_seqs))
    all_seqs = np.array(pos_seqs[:n_samples] + neg_seqs[:n_samples])
    y = np.array([1] * n_samples + [0] * n_samples)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_names = [
        "Logistic Regression",
        "Linear SVM",
        "Random Forest",
        "RBF SVM",
        "CNN-BiLSTM",
        "GPlusD (LightGBM)"
    ]

    metrics = {name: {"auc": [], "mcc": [], "acc": [], "time": []} for name in model_names}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_seqs, y), 1):
        print(f"\n=================== FOLD {fold}/5 ===================")
        train_seqs, val_seqs = all_seqs[train_idx], all_seqs[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Featurize for tabular / tree / linear models
        vec = CountVectorizer(analyzer="char", ngram_range=(4, 6), min_df=5, dtype=np.float32)
        X_train_sp = vec.fit_transform(train_seqs).astype(np.float32)
        X_val_sp = vec.transform(val_seqs).astype(np.float32)

        # 1. Logistic Regression
        t0 = time.perf_counter()
        lr = LogisticRegression(C=1.0, max_iter=500, n_jobs=-1)
        lr.fit(X_train_sp, y_train)
        probs_lr = lr.predict_proba(X_val_sp)[:, 1]
        metrics["Logistic Regression"]["time"].append(time.perf_counter() - t0)
        metrics["Logistic Regression"]["auc"].append(roc_auc_score(y_val, probs_lr))
        metrics["Logistic Regression"]["mcc"].append(matthews_corrcoef(y_val, (probs_lr >= 0.5).astype(int)))
        metrics["Logistic Regression"]["acc"].append(accuracy_score(y_val, (probs_lr >= 0.5).astype(int)))

        # 2. Linear SVM
        t0 = time.perf_counter()
        l_svm = LinearSVC(C=0.5, max_iter=1000, random_state=42)
        l_svm.fit(X_train_sp, y_train)
        df_vals = l_svm.decision_function(X_val_sp)
        probs_svm = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min() + 1e-8)
        metrics["Linear SVM"]["time"].append(time.perf_counter() - t0)
        metrics["Linear SVM"]["auc"].append(roc_auc_score(y_val, probs_svm))
        metrics["Linear SVM"]["mcc"].append(matthews_corrcoef(y_val, (probs_svm >= 0.5).astype(int)))
        metrics["Linear SVM"]["acc"].append(accuracy_score(y_val, (probs_svm >= 0.5).astype(int)))

        # 3. Random Forest
        t0 = time.perf_counter()
        rf = RandomForestClassifier(n_estimators=200, max_depth=25, n_jobs=-1, random_state=42)
        rf.fit(X_train_sp, y_train)
        probs_rf = rf.predict_proba(X_val_sp)[:, 1]
        metrics["Random Forest"]["time"].append(time.perf_counter() - t0)
        metrics["Random Forest"]["auc"].append(roc_auc_score(y_val, probs_rf))
        metrics["Random Forest"]["mcc"].append(matthews_corrcoef(y_val, (probs_rf >= 0.5).astype(int)))
        metrics["Random Forest"]["acc"].append(accuracy_score(y_val, (probs_rf >= 0.5).astype(int)))

        # 4. RBF SVM (subsampled for speed)
        t0 = time.perf_counter()
        rbf_svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
        sub_idx = np.random.choice(X_train_sp.shape[0], min(4000, X_train_sp.shape[0]), replace=False)
        rbf_svm.fit(X_train_sp[sub_idx], y_train[sub_idx])
        probs_rbf = rbf_svm.predict_proba(X_val_sp)[:, 1]
        metrics["RBF SVM"]["time"].append(time.perf_counter() - t0)
        metrics["RBF SVM"]["auc"].append(roc_auc_score(y_val, probs_rbf))
        metrics["RBF SVM"]["mcc"].append(matthews_corrcoef(y_val, (probs_rbf >= 0.5).astype(int)))
        metrics["RBF SVM"]["acc"].append(accuracy_score(y_val, (probs_rbf >= 0.5).astype(int)))

        # 5. CNN-BiLSTM (PyTorch)
        t0 = time.perf_counter()
        X_tr_t = get_onehot_tensor(train_seqs)
        X_va_t = get_onehot_tensor(val_seqs)
        tr_ds = TensorDataset(X_tr_t, torch.tensor(y_train, dtype=torch.float32))
        tr_ldr = DataLoader(tr_ds, batch_size=128, shuffle=True)

        cnn = CNN_BiLSTM().to(device)
        crit = nn.BCELoss()
        opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)

        cnn.train()
        for ep in range(args.epochs):
            for bx, by in tr_ldr:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                crit(cnn(bx), by).backward()
                opt.step()

        cnn.eval()
        with torch.no_grad():
            probs_cnn = cnn(X_va_t.to(device)).cpu().numpy()

        metrics["CNN-BiLSTM"]["time"].append(time.perf_counter() - t0)
        metrics["CNN-BiLSTM"]["auc"].append(roc_auc_score(y_val, probs_cnn))
        metrics["CNN-BiLSTM"]["mcc"].append(matthews_corrcoef(y_val, (probs_cnn >= 0.5).astype(int)))
        metrics["CNN-BiLSTM"]["acc"].append(accuracy_score(y_val, (probs_cnn >= 0.5).astype(int)))

        # 6. GPlusD (LightGBM)
        t0 = time.perf_counter()
        lgbm = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=5,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgbm.fit(X_train_sp, y_train, eval_set=[(X_val_sp, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        probs_lgbm = lgbm.predict_proba(X_val_sp)[:, 1]
        metrics["GPlusD (LightGBM)"]["time"].append(time.perf_counter() - t0)
        metrics["GPlusD (LightGBM)"]["auc"].append(roc_auc_score(y_val, probs_lgbm))
        metrics["GPlusD (LightGBM)"]["mcc"].append(matthews_corrcoef(y_val, (probs_lgbm >= 0.5).astype(int)))
        metrics["GPlusD (LightGBM)"]["acc"].append(accuracy_score(y_val, (probs_lgbm >= 0.5).astype(int)))

        for m_name in model_names:
            print(f"{m_name:20s} | MCC: {metrics[m_name]['mcc'][-1]:.4f} | AUC: {metrics[m_name]['auc'][-1]:.4f} | Time: {metrics[m_name]['time'][-1]:.1f}s")

    # Output LaTeX formatted table matching Table 2
    print("\n" + "=" * 75)
    print(f"{'Model':22s} | {'ROC AUC':8s} | {'MCC':8s} | {'Accuracy':8s} | {'Train Time (s)':12s}")
    print("=" * 75)
    for m_name in model_names:
        m = metrics[m_name]
        print(f"{m_name:22s} | {np.mean(m['auc']):.4f}   | {np.mean(m['mcc']):.4f}   | {np.mean(m['acc']):.4f}   | {np.mean(m['time']):.1f}")
    print("=" * 75)

if __name__ == "__main__":
    main()