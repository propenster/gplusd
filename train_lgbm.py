import argparse
import joblib
from pathlib import Path
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef, roc_auc_score
from dataset_builder import prepare_dataset

def train_lightgbm(data_path, output_dir="output", k_size=3):
    exp_folder = Path(f"{output_dir}")
    exp_folder.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Data
    df = prepare_dataset(data_path, k_size=k_size)
    X = df.drop(columns=['label'])
    y = df['label']
    
    print(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features.")

    # 2. Setup Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42
    }

    best_mcc = -1
    fold_metrics = []

    print("\nStarting 5-Fold Cross Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params, 
            train_data, 
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        # Evaluate Fold
        val_preds_proba = model.predict(X_val)
        val_preds = (val_preds_proba >= 0.5).astype(int)
        
        precision = precision_score(y_val, val_preds, zero_division=0)
        recall = recall_score(y_val, val_preds, zero_division=0)
        mcc = matthews_corrcoef(y_val, val_preds)
        auc = roc_auc_score(y_val, val_preds_proba)
        
        print(f"Fold {fold}: AUC = {auc:.4f} | MCC = {mcc:.4f} | P = {precision:.4f} | R = {recall:.4f}")
        fold_metrics.append({'auc': auc, 'mcc': mcc, 'precision': precision, 'recall': recall})
        
        # Save best model based on MCC
        if mcc > best_mcc:
            best_mcc = mcc
            joblib.dump(model, exp_folder / "best_model.pkl")
            # We also save the feature columns to ensure alignment during inference
            joblib.dump(list(X.columns), exp_folder / "feature_columns.pkl")
            print(f"  -> New best model saved (MCC: {mcc:.4f})")

    # Log overall metrics
    mean_mcc = np.mean([m['mcc'] for m in fold_metrics])
    mean_auc = np.mean([m['auc'] for m in fold_metrics])
    print(f"\nTraining Complete. Average AUC: {mean_auc:.4f} | Average MCC: {mean_mcc:.4f}")

    with open(exp_folder / "log.txt", "w") as f:
        f.write(f"5-Fold Cross Validation Results:\n")
        f.write(f"Mean AUC: {mean_auc:.4f}\n")
        f.write(f"Mean MCC: {mean_mcc:.4f}\n")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, required=True, help="path to dataset (txt file)")
    parser.add_argument("-o", "--output-dir", type=str, default="test", help="output subfolder name")
    parser.add_argument("-k", "--k-size", type=int, default=3, help="k-mer size for De Bruijn graph")
    # parser.add_argument("-p", "--model-prefix", type=str, help="prefix for naming the model e.g gminus gplus")

    args = parser.parse_args()

    train_lightgbm(data_path=args.data, output_dir=args.output_dir, k_size=args.k_size)