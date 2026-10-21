import os
import re
import subprocess
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb

def prepare_fasta(pos_file, fasta_out="temp_positives.fasta"):
    with open(pos_file, "r") as f_in, open(fasta_out, "w") as f_out:
        for idx, line in enumerate(f_in):
            seq = line.strip().upper()
            if len(seq) == 81:
                f_out.write(f">seq_{idx}\n{seq}\n")
    return fasta_out

def run_cdhit(fasta_in, prefix="cdhit_clusters", c=0.80, n=5):
    clstr_file = f"{prefix}.clstr"
    if os.path.exists(clstr_file):
        print(f"Found existing clustering file: {clstr_file}. Skipping CD-HIT execution.")
        return clstr_file

    cmd = [
        "cd-hit-est",
        "-i", fasta_in,
        "-o", prefix,
        "-c", str(c),
        "-n", str(n),
        "-d", "0",
        "-M", "16000",
        "-T", "0"
    ]
    print(f"Running CD-HIT-EST: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return clstr_file

def parse_cdhit_clusters(clstr_file, total_samples):
    cluster_map = {}
    current_cluster = None
    with open(clstr_file, "r") as f:
        for line in f:
            if line.startswith(">Cluster"):
                current_cluster = int(line.strip().split()[1])
            else:
                match = re.search(r">seq_(\d+)\.\.\.", line)
                if match:
                    idx = int(match.group(1))
                    cluster_map[idx] = current_cluster
    return [cluster_map.get(i, i) for i in range(total_samples)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", default="data/gplusd_positive.txt")
    parser.add_argument("--neg", default="data/gplusd_negative.txt")
    parser.add_argument("--c", type=float, default=0.80)
    args = parser.parse_args()

    # 1. Load sequence pairs
    with open(args.pos, "r") as f:
        pos_seqs = [l.strip().upper() for l in f if len(l.strip()) == 81]
    with open(args.neg, "r") as f:
        neg_seqs = [l.strip().upper() for l in f if len(l.strip()) == 81]

    n_samples = min(len(pos_seqs), len(neg_seqs))
    pos_seqs = pos_seqs[:n_samples]
    neg_seqs = neg_seqs[:n_samples]
    print(f"Loaded {n_samples} balanced positive and negative pairs (81 bp each).")

    # 2. Cluster positive sequences (or reuse existing cdhit_clusters.clstr)
    clstr_file = f"cdhit_clusters.clstr"
    if not os.path.exists(clstr_file):
        fasta_file = prepare_fasta(args.pos)
        clstr_file = run_cdhit(fasta_file, c=args.c)
        if os.path.exists(fasta_file):
            os.remove(fasta_file)
    else:
        print(f"Reusing existing {clstr_file}...")

    pos_clusters = parse_cdhit_clusters(clstr_file, n_samples)

    all_seqs = np.array(pos_seqs + neg_seqs)
    y = np.array([1] * n_samples + [0] * n_samples)
    groups = np.array(pos_clusters + pos_clusters)

    # 3. Group-stratified 5-fold cross validation
    sgkf = StratifiedGroupKFold(n_splits=5)
    metrics = {"acc": [], "prec": [], "rec": [], "mcc": [], "roc_auc": []}

    print("\nStarting 5-Fold Cluster-Stratified Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(all_seqs, y, groups=groups), 1):
        # Explicitly set dtype=np.float32 to prevent LightGBM int64 TypeError
        vec = CountVectorizer(analyzer="char", ngram_range=(4, 6), min_df=5, dtype=np.float32)
        X_train = vec.fit_transform(all_seqs[train_idx]).astype(np.float32)
        X_val = vec.transform(all_seqs[val_idx]).astype(np.float32)

        y_train, y_val = y[train_idx], y[val_idx]

        clf = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=5,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )

        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        metrics["acc"].append(accuracy_score(y_val, preds))
        metrics["prec"].append(precision_score(y_val, preds, zero_division=0))
        metrics["rec"].append(recall_score(y_val, preds, zero_division=0))
        metrics["mcc"].append(matthews_corrcoef(y_val, preds))
        metrics["roc_auc"].append(roc_auc_score(y_val, probs))

        print(f"Fold {fold}: AUC = {metrics['roc_auc'][-1]:.4f} | MCC = {metrics['mcc'][-1]:.4f} | Acc = {metrics['acc'][-1]:.4f}")

    print("\n" + "=" * 55)
    print(f"CLUSTER-BASED CV RESULTS (CD-HIT c={args.c}):")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"{k.upper():8s}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    print("=" * 55)

if __name__ == "__main__":
    main()