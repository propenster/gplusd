import argparse
import joblib
import pandas as pd
from pathlib import Path
from dataset_builder import extract_promoter_features

def infer(data_path, model_dir):
    model = joblib.load(Path(model_dir) / "best_model.pkl")
    feature_cols = joblib.load(Path(model_dir) / "feature_columns.pkl")
    
    with open(data_path, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
        
    print(f"Running inference on {len(sequences)} sequences...")
    
    # Extract features for new sequences
    features_list = [extract_promoter_features(seq) for seq in sequences]
    df = pd.DataFrame(features_list)
    
    # Ensure columns match training exactly
    df = df.reindex(columns=feature_cols, fill_value=0)
    
    # Predict
    probabilities = model.predict(df)
    predictions = (probabilities >= 0.5).astype(int)
    
    with open("infer_results.txt", "w") as f:
        for seq, prob, pred in zip(sequences, probabilities, predictions):
            f.write(f"Seq: {seq[:10]}... | Prob: {prob:.4f} | Class: {pred}\n")
            
    print("Results saved to infer_results.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, required=True, help="path to sequences to predict (txt file)")
    parser.add_argument("-m", "--model-dir", type=str, required=True, help="directory containing best_model.pkl")
    args = parser.parse_args()

    infer(args.data, args.model_dir)