# Add PyTorch imports at the top
import torch
import torch.nn as nn
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import logomaker
from dataset_builder import extract_promoter_features
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
        return self.sigmoid(self.fc(self.dropout2(x[:, -1, :]))).squeeze()

def get_onehot_tensor(seqs):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    tensors = [np.array([mapping.get(b, [0.25]*4) for b in s]).T for s in seqs]
    return torch.tensor(np.array(tensors), dtype=torch.float32)



def plot_roc_pr_curves(y_true, y_proba, output_dir):
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AEI-LGBM (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.set_title('ROC Curve (External Benchmark)', fontweight='bold')
    ax1.legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    ax2.plot(recall, precision, color='green', lw=2, label=f'AEI-LGBM (AP = {ap:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision-Recall Curve (External Benchmark)', fontweight='bold')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'roc_pr_curves.pdf', dpi=400)
    print("Saved ROC and PR curves.")



def generate_3panel_logos(pos_train_file, bio_neg_file, aei_neg_file, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def get_info_matrix(file_path):
        with open(file_path, 'r') as f:
            seqs = [line.strip().upper() for line in f if len(line.strip()) == 81]
        counts = pd.DataFrame(0, index=range(81), columns=['A', 'C', 'G', 'T'])
        for seq in seqs:
            for i, base in enumerate(seq[:81]):
                if base in counts.columns:
                    counts.loc[i, base] += 1
        prob_matrix = counts.div(counts.sum(axis=1), axis=0)
        return logomaker.transform_matrix(prob_matrix, from_type='probability', to_type='information')

    # 1. Positive Training Logo
    fig, ax1 = plt.subplots(figsize=(10, 3))
    logomaker.Logo(get_info_matrix(pos_train_file), ax=ax1, color_scheme='classic')
    ax1.set_title('True Gram-Positive Promoters (Positive Set)', fontweight='bold')
    ax1.set_ylabel('bits', fontweight='bold')
    ax1.set_ylim([0, 2.0])
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'logo_1_positive.pdf', dpi=600)
    plt.close()

    # 2. Biological Genomic Background Logo (CNNPromoter)
    fig, ax2 = plt.subplots(figsize=(10, 3))
    logomaker.Logo(get_info_matrix(bio_neg_file), ax=ax2, color_scheme='classic')
    ax2.set_title('Biological Genomic Background (Traditional Negatives)', fontweight='bold')
    ax2.set_ylabel('bits', fontweight='bold')
    ax2.set_ylim([0, 2.0])
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'logo_2_bio_negative.pdf', dpi=600)
    plt.close()

    # 3. AEI Engineered Background Logo
    fig, ax3 = plt.subplots(figsize=(10, 3))
    logomaker.Logo(get_info_matrix(aei_neg_file), ax=ax3, color_scheme='classic')
    ax3.set_title('De Bruijn AEI Engineered Negatives (Proposed)', fontweight='bold')
    ax3.set_ylabel('bits', fontweight='bold')
    ax3.set_xlabel('Sequence Position', fontweight='bold')
    ax3.set_ylim([0, 2.0])
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'logo_3_aei_negative.pdf', dpi=600)
    plt.close()
    print("Saved 3-panel logos separately.")



def generate_comparative_logo(pos_file, neg_file, output_dir):
    with open(pos_file, 'r') as f:
        pos_seqs = [line.strip().upper() for line in f if len(line.strip()) == 81]
    with open(neg_file, 'r') as f:
        neg_seqs = [line.strip().upper() for line in f if len(line.strip()) == 81]

    def get_info_matrix(seqs):
        counts = pd.DataFrame(0, index=range(81), columns=['A', 'C', 'G', 'T'])
        for seq in seqs:
            for i, base in enumerate(seq[:81]):
                if base in counts.columns:
                    counts.loc[i, base] += 1
        prob_matrix = counts.div(counts.sum(axis=1), axis=0)
        return logomaker.transform_matrix(prob_matrix, from_type='probability', to_type='information')

    pos_info = get_info_matrix(pos_seqs)
    neg_info = get_info_matrix(neg_seqs)

    # Positive Logo
    fig, ax1 = plt.subplots(figsize=(10, 3))
    logomaker.Logo(pos_info, ax=ax1, color_scheme='classic')
    ax1.set_title('Gram-Positive Promoters (Positive Training Set)', fontweight='bold')
    ax1.set_ylabel('bits', fontweight='bold')
    ax1.set_ylim([0, 2.0])
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'logo_positive.pdf', dpi=600)
    plt.close()

    # Negative Logo
    fig, ax2 = plt.subplots(figsize=(10, 3))
    logomaker.Logo(neg_info, ax=ax2, color_scheme='classic')
    ax2.set_title('De Bruijn AEI Non-Promoters (Negative Training Set)', fontweight='bold')
    ax2.set_ylabel('bits', fontweight='bold')
    ax2.set_xlabel('Sequence Position', fontweight='bold')
    ax2.set_ylim([0, 2.0])
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'logo_negative.pdf', dpi=600)
    plt.close()

def plot_true_positive_logo(tp_seqs, output_dir):
    if not tp_seqs:
        print("No True Positives found to plot logo.")
        return
        
    counts = pd.DataFrame(0, index=range(81), columns=['A', 'C', 'G', 'T'])
    for seq in tp_seqs:
        for i, base in enumerate(seq):
            if base in counts.columns:
                counts.loc[i, base] += 1
                
    prob_matrix = counts.div(counts.sum(axis=1), axis=0)
    info_matrix = logomaker.transform_matrix(prob_matrix, from_type='probability', to_type='information')
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    logomaker.Logo(info_matrix, ax=ax, color_scheme='classic')
    
    # Approximate regions for -35 and -10 boxes based on -60 to +20 alignment
    ax.axvspan(20, 26, color='red', alpha=0.1, label='-35 Box Region')
    ax.axvspan(44, 50, color='blue', alpha=0.1, label='-10 Box Region')
    
    ax.set_title("Sequence Logo of External True Positives (Successfully Predicted)", fontweight='bold', fontsize=16)
    ax.set_ylabel("Information (bits)", fontweight='bold', fontsize=12)
    ax.set_xlabel("Position (bp)", fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'true_positive_sequence_logo.pdf', dpi=400)
    print("Saved True Positive Sequence Logo.")

# def main(ext_pos, ext_neg, train_pos, train_neg, model_dir, output_dir):
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
    
#     model = joblib.load(Path(model_dir) / "best_model.pkl")
#     feature_cols = joblib.load(Path(model_dir) / "feature_columns.pkl")
    
#     with open(ext_pos, 'r') as f:
#         pos_seqs = [line.strip() for line in f if len(line.strip()) == 81]
#     with open(ext_neg, 'r') as f:
#         neg_seqs = [line.strip() for line in f if len(line.strip()) == 81]
        
#     all_seqs = pos_seqs + neg_seqs
#     y_true = np.array([1]*len(pos_seqs) + [0]*len(neg_seqs))
    
#     features_list = [extract_promoter_features(seq) for seq in all_seqs]
#     df = pd.DataFrame(features_list).reindex(columns=feature_cols, fill_value=0)
    
#     y_proba = model.predict(df)
#     # y_pred = (y_proba >= 0.5).astype(int)
#     # Calculate Optimal Threshold dynamically for the plots
#     fpr, tpr, thresholds = roc_curve(y_true, y_proba)
#     optimal_idx = np.argmax(tpr - fpr)
#     optimal_threshold = thresholds[optimal_idx]
#     print(f"Plotting using Optimal Threshold: {optimal_threshold:.4f}")
#     y_pred = (y_proba >= optimal_threshold).astype(int)
    
#     # Isolate the sequences that were Positive AND the model predicted Positive
#     tp_seqs = [pos_seqs[i] for i in range(len(pos_seqs)) if y_pred[i] == 1]
    
#     # 1. Evaluate External Benchmark
#     # plot_roc_pr_curves(y_true, y_proba, output_dir)
#     plot_roc_curve(y_true, y_proba, output_dir)
#     plot_pr_curve(y_true, y_proba, output_dir)
#     # 2. Generate True Positive Logo
#     plot_true_positive_logo(tp_seqs, output_dir)
#     # plot_negative_logo(neg_seqs, output_dir)
#     # 3. Generate Comparative Training Logo
#     if train_pos and train_neg:
#         # generate_comparative_logo(train_pos, train_neg, output_dir)
#         generate_3panel_logos(train_pos, ext_neg, train_neg, output_dir)
        
#     print(f"Successfully generated all paper figures in '{output_dir}'.")

def plot_roc_curve(y_true, y_proba, output_dir):
    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'AEI-LightGBM (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    plt.title('ROC Curve (External Benchmark)', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'roc_curve.pdf', dpi=600)
    plt.close()

def plot_pr_curve(y_true, y_proba, output_dir):
    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    
    plt.plot(recall, precision, color='green', lw=2.5, label=f'AEI-LightGBM (AP = {ap:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontweight='bold', fontsize=12)
    plt.ylabel('Precision', fontweight='bold', fontsize=12)
    plt.title('Precision-Recall Curve', fontweight='bold', fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'pr_curve.pdf', dpi=600)
    plt.close()






def main(ext_pos, ext_neg, train_pos, train_neg, model_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # LOAD PYTORCH MODEL INSTEAD OF JOBLIB
    model = CNN_BiLSTM()
    model.load_state_dict(torch.load(Path(model_dir) / "best_model.pt"))
    model.eval()
    
    with open(ext_pos, 'r') as f:
        pos_seqs = [line.strip() for line in f if len(line.strip()) == 81]
    with open(ext_neg, 'r') as f:
        neg_seqs = [line.strip() for line in f if len(line.strip()) == 81]
        
    all_seqs = pos_seqs + neg_seqs
    y_true = np.array([1]*len(pos_seqs) + [0]*len(neg_seqs))
    
    # INFERENCE VIA PYTORCH TENSOR
    X_tensor = get_onehot_tensor(all_seqs)
    with torch.no_grad():
        y_proba = model(X_tensor).numpy()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    tp_seqs = [pos_seqs[i] for i in range(len(pos_seqs)) if y_pred[i] == 1]
    
    # (Keep the rest of your plotting function calls the same)
    plot_roc_curve(y_true, y_proba, output_dir)
    plot_pr_curve(y_true, y_proba, output_dir)
    plot_true_positive_logo(tp_seqs, output_dir)
    if train_pos and train_neg:
        generate_3panel_logos(train_pos, ext_neg, train_neg, output_dir)
        
        
        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ext-pos", type=str, default="data/dbtbs_positives.txt")
    parser.add_argument("--ext-neg", type=str, default="data/dbtbs_negatives.txt")
    parser.add_argument("--train-pos", type=str, help="e.g., data/gplus.txt")
    parser.add_argument("--train-neg", type=str, help="e.g., data/gplus_negative.txt")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="figs")
    # parser.add_argument("--model-prefix", type=str, help="prefix for naming the model e.g gminus gplus")
    
    args = parser.parse_args()

    main(args.ext_pos, args.ext_neg, args.train_pos, args.train_neg, args.model_dir, args.out_dir)