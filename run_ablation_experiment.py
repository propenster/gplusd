import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, matthews_corrcoef, precision_score, recall_score
from dataset_builder import extract_promoter_features
from generate_negative_sequences import generate_adversarial_negative

def random_shuffle_seq(seq):
    """Method 1: Complete mononucleotide random shuffle."""
    seq_list = list(seq)
    random.shuffle(seq_list)
    return "".join(seq_list)

def block_shuffle_seq(seq, num_blocks=10, shuffle_ratio=0.6):
    """Method 2: DeePromoter-style block segmentation and partial shuffling."""
    block_size = len(seq) // num_blocks
    blocks = [seq[i*block_size:(i+1)*block_size] for i in range(num_blocks)]
    
    num_to_shuffle = int(num_blocks * shuffle_ratio)
    indices_to_shuffle = random.sample(range(num_blocks), num_to_shuffle)
    
    shuffled_sub_blocks = [blocks[i] for i in indices_to_shuffle]
    random.shuffle(shuffled_sub_blocks)
    
    for idx, new_block in zip(indices_to_shuffle, shuffled_sub_blocks):
        blocks[idx] = new_block
        
    return "".join(blocks)

def build_dataset(pos_seqs, neg_gen_func):
    neg_seqs = [neg_gen_func(s) for s in pos_seqs]
    all_seqs = pos_seqs + neg_seqs
    y = np.array([1]*len(pos_seqs) + [0]*len(neg_seqs))
    
    features = [extract_promoter_features(s) for s in all_seqs]
    df = pd.DataFrame(features)
    return df, y

def run_ablation(train_pos_path, ext_pos_path, ext_neg_path, output_dir="docs/ablation"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(42)
    
    with open(train_pos_path, 'r') as f:
        train_pos = [l.strip().upper() for l in f if len(l.strip()) == 81]
    with open(ext_pos_path, 'r') as f:
        ext_pos = [l.strip().upper() for l in f if len(l.strip()) == 81]
    with open(ext_neg_path, 'r') as f:
        ext_neg = [l.strip().upper() for l in f if len(l.strip()) == 81]

    methods = {
        'AEI (Proposed, k=5)': lambda s: generate_adversarial_negative(s, k=5) or s,
        'Random Mononucleotide Shuffle': random_shuffle_seq,
        'Block Permutation (DeePromoter-style)': lambda s: block_shuffle_seq(s, num_blocks=10, shuffle_ratio=0.6)
    }

    ext_all = ext_pos + ext_neg
    y_ext = np.array([1]*len(ext_pos) + [0]*len(ext_neg))
    X_ext = pd.DataFrame([extract_promoter_features(s) for s in ext_all])

    results = []
    
    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    for name, gen_func in methods.items():
        print(f"Running ablation for method: {name}...")
        X_train, y_train = build_dataset(train_pos, gen_func)
        
        # Align columns
        X_ext_aligned = X_ext.reindex(columns=X_train.columns, fill_value=0)
        
        # Train LightGBM model
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1)
        clf.fit(X_train, y_train)
        
        # Evaluate on unseen External Benchmark
        y_proba = clf.predict_proba(X_ext_aligned)[:, 1]
        
        # Calculate curves
        fpr, tpr, thresholds = roc_curve(y_ext, y_proba)
        roc_auc = auc(fpr, tpr)
        
        precision_vals, recall_vals, _ = precision_recall_curve(y_ext, y_proba)
        ap = average_precision_score(y_ext, y_proba)
        
        # Optimal threshold metrics
        optimal_idx = np.argmax(tpr - fpr)
        opt_thresh = thresholds[optimal_idx]
        y_pred = (y_proba >= opt_thresh).astype(int)
        
        mcc = matthews_corrcoef(y_ext, y_pred)
        prec = precision_score(y_ext, y_pred)
        rec = recall_score(y_ext, y_pred)
        
        results.append({
            'Strategy': name,
            'AUC': roc_auc,
            'AP': ap,
            'MCC': mcc,
            'Precision': prec,
            'Recall': rec,
            'Optimal Thresh': opt_thresh
        })
        
        # Plot ROC & PR
        ax1.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        ax2.plot(recall_vals, precision_vals, lw=2, label=f'{name} (AP = {ap:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.set_title('ROC Curve Comparison Across Negative Controls', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)

    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision-Recall Curve Comparison Across Controls', fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'negative_sampling_ablation.png', dpi=600)
    
    df_res = pd.DataFrame(results)
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS (EXTERNAL DBTBS BENCHMARK)")
    print("="*60)
    print(df_res.to_string(index=False))
    return df_res

if __name__ == '__main__':
    run_ablation('data/gplus.txt', 'data/dbtbs_positives.txt', 'data/dbtbs_negatives.txt')