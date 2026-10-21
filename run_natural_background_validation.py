import argparse
import random
import numpy as np
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
import lightgbm as lgb

def extract_natural_negatives(genome_fasta, n_samples=18500, seq_len=81):
    """Samples genuine non-promoter genomic sequences from B. subtilis."""
    record = next(SeqIO.parse(genome_fasta, "fasta"))
    genome = str(record.seq).upper()
    genome_len = len(genome)

    negatives = set()
    attempts = 0
    max_attempts = n_samples * 10

    # Ensure random sampling covers both strands across the genome
    while len(negatives) < n_samples and attempts < max_attempts:
        attempts += 1
        pos = random.randint(0, genome_len - seq_len)
        chunk = genome[pos : pos + seq_len]
        if "N" not in chunk:
            negatives.add(chunk)

    return list(negatives)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pos", default="data/gplusd_positive.txt")
    parser.add_argument("--train-neg", default="data/gplusd_negative.txt")
    parser.add_argument("--genome", default="bsubtilis_genome.fasta")
    parser.add_argument("--n-samples", type=int, default=18500)
    args = parser.parse_args()

    # 1. Load training sequences (GPlusD trained on composition-matched Eulerian pairs)
    with open(args.train_pos, "r") as f:
        train_pos = [l.strip().upper() for l in f if len(l.strip()) == 81]
    with open(args.train_neg, "r") as f:
        train_neg = [l.strip().upper() for l in f if len(l.strip()) == 81]

    # 2. Extract natural genomic negatives
    print(f"Sampling {args.n_samples} genuine biological non-promoter windows from {args.genome}...")
    natural_neg = extract_natural_negatives(args.genome, n_samples=min(args.n_samples, len(train_pos)))
    eval_pos = train_pos[:len(natural_neg)]

    # 3. Train model on composition-matched sets
    print("Training GPlusD on Eulerian composition-matched training set...")
    vec = CountVectorizer(analyzer="char", ngram_range=(4, 6), min_df=5, dtype=np.float32)
    X_train = vec.fit_transform(train_pos + train_neg).astype(np.float32)
    y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))

    clf = lgb.LGBMClassifier(
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
    clf.fit(X_train, y_train)

    # 4. Evaluate against genuine natural genomic negatives
    print("Evaluating GPlusD against genuine genomic background...")
    X_eval = vec.transform(eval_pos + natural_neg)
    y_eval = np.array([1] * len(eval_pos) + [0] * len(natural_neg))

    probs = clf.predict_proba(X_eval)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_eval, probs)
    mcc = matthews_corrcoef(y_eval, preds)
    acc = accuracy_score(y_eval, preds)
    prec = precision_score(y_eval, preds, zero_division=0)
    rec = recall_score(y_eval, preds, zero_division=0)

    print("\n" + "=" * 55)
    print("VALIDATION AGAINST NATURAL GENOMIC BACKGROUND:")
    print("=" * 55)
    print(f"ROC AUC:   {auc:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("=" * 55)

if __name__ == "__main__":
    main()