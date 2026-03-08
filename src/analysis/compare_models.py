import torch
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.models.graph_model import GATLinkPredictor
from src.analysis.explainability import PPIExplainer
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def find_optimal_threshold(y_true, y_prob, method="f1"):
    """
    Sweep thresholds from 0.1 to 0.9 and find the optimal one.
    method: 'f1' (maximize F1) or 'youden' (maximize Youden's index = TPR - FPR)
    """
    best_thresh = 0.5
    best_score = -1
    
    for thresh in np.arange(0.1, 0.91, 0.01):
        y_pred = (y_prob > thresh).astype(int)
        if method == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif method == "youden":
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            score = tpr - fpr  # Youden's J
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh, best_score

def evaluate_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating models on {device}...")

    # Load test data
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        print(f"Test data not found at {test_path}")
        return

    test_df = pd.read_csv(test_path)
    print(f"Loaded {len(test_df)} test samples.")

    # Load embeddings and mapping
    emb_path = PROCESSED_DATA_DIR / "embeddings.pt"
    map_path = PROCESSED_DATA_DIR / "ppi_graph_mapping.pt"
    graph_data_path = PROCESSED_DATA_DIR / "ppi_graph.pt"
    
    if not emb_path.exists() or not map_path.exists() or not graph_data_path.exists():
        print("Required processed data (embeddings, mapping, graph) missing.")
        return
        
    embeddings = torch.load(emb_path, map_location=device, weights_only=False)
    node_mapping = torch.load(map_path, map_location=device, weights_only=False)
    graph_data = torch.load(graph_data_path, map_location=device, weights_only=False)

    # Filter test_df to valid entries
    filtered_df = test_df[
        test_df["protein1"].isin(embeddings) & 
        test_df["protein2"].isin(embeddings) &
        test_df["protein1"].isin(node_mapping) &
        test_df["protein2"].isin(node_mapping)
    ].copy()

    # Load Models
    seq_path = PROJECT_ROOT / "models" / "sequence_model_best.pth"
    graph_model_path = PROJECT_ROOT / "models" / "graph_model_best.pth"
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"

    sample_emb = next(iter(embeddings.values()))
    # Use the last dimension for input_dim (handles both 1D and 2D embeddings)
    input_dim = sample_emb.shape[-1] if sample_emb.dim() > 1 else sample_emb.shape[0]
    seq_model = SequencePPIModel(input_dim=input_dim, hidden_dim=256, dropout=0.3).to(device)
    if seq_path.exists():
         seq_model.load_state_dict(torch.load(seq_path, map_location=device))
    seq_model.eval()

    # Updated GAT: hidden_channels=128, num_layers=3
    graph_model = GATLinkPredictor(in_channels=graph_data.x.shape[1], hidden_channels=128, num_layers=3).to(device)
    if graph_model_path.exists():
         graph_model.load_state_dict(torch.load(graph_model_path, map_location=device))
    graph_model.eval()

    ensemble_model = None
    if ensemble_path.exists():
         ensemble_model = joblib.load(ensemble_path)

    # Prepare batch data
    batch_emb1 = []
    batch_emb2 = []
    g_src = []
    g_dst = []
    labels = []

    for _, row in filtered_df.iterrows():
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        e1 = embeddings[p1]
        e2 = embeddings[p2]
        # Extract CLS token if embeddings are 2D (sequence_length x dim)
        batch_emb1.append(e1[0] if e1.dim() > 1 else e1)
        batch_emb2.append(e2[0] if e2.dim() > 1 else e2)
        g_src.append(node_mapping[p1])
        g_dst.append(node_mapping[p2])
        labels.append(label)

    labels = np.array(labels)

    # Predict Sequence
    batch_emb1 = torch.stack(batch_emb1).to(device)
    batch_emb2 = torch.stack(batch_emb2).to(device)
    
    seq_preds = []
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(batch_emb1), batch_size):
            out = seq_model(batch_emb1[i:i+batch_size], batch_emb2[i:i+batch_size])
            # Apply sigmoid to raw logits
            probs = torch.sigmoid(out)
            seq_preds.extend(probs.cpu().numpy().flatten())
    seq_preds = np.array(seq_preds)

    # Predict Graph — apply sigmoid to raw logits
    g_edge_label_index = torch.tensor([g_src, g_dst], dtype=torch.long).to(device)
    graph_preds = []
    with torch.no_grad():
        batch_size = 10000
        for i in range(0, g_edge_label_index.size(1), batch_size):
            chunk = g_edge_label_index[:, i:i+batch_size]
            out = graph_model(graph_data.x, graph_data.edge_index, chunk)
            # Apply sigmoid to raw logits
            probs = torch.sigmoid(out)
            graph_preds.extend(probs.cpu().numpy().flatten())
    graph_preds = np.array(graph_preds)

    # Predict Ensemble — with enhanced features
    ens_preds = None
    if ensemble_model:
        # Enhanced features: [seq, gat, |seq-0.5|, |gat-0.5|]
        conf_seq = np.abs(seq_preds - 0.5)
        conf_gat = np.abs(graph_preds - 0.5)
        X = np.column_stack((seq_preds, graph_preds, conf_seq, conf_gat))
        ens_preds = ensemble_model.predict_proba(X)[:, 1]

    # Calculate Metrics (default threshold 0.5)
    def calc_metrics(y_true, y_prob, threshold=0.5):
        y_pred = (y_prob > threshold).astype(int)
        return [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0),
            roc_auc_score(y_true, y_prob),
            average_precision_score(y_true, y_prob)
        ]

    # === Standard Results (threshold=0.5) ===
    results = []
    results.append(["ESM-MLP"] + calc_metrics(labels, seq_preds))
    results.append(["GAT"] + calc_metrics(labels, graph_preds))
    if ens_preds is not None:
         results.append(["Ensemble"] + calc_metrics(labels, ens_preds))

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print("\n=== Results (threshold=0.5) ===")
    print(tabulate(results, headers=headers, floatfmt=".4f", tablefmt="grid"))

    # === Threshold Tuning ===
    print("\n=== Optimal Threshold Tuning (F1-maximizing) ===")
    tuning_results = []
    
    for name, preds in [("ESM-MLP", seq_preds), ("GAT", graph_preds)]:
        best_t, best_f1 = find_optimal_threshold(labels, preds, method="f1")
        tuned_metrics = calc_metrics(labels, preds, threshold=best_t)
        tuning_results.append([name, best_t] + tuned_metrics)
    
    if ens_preds is not None:
        best_t, best_f1 = find_optimal_threshold(labels, ens_preds, method="f1")
        tuned_metrics = calc_metrics(labels, ens_preds, threshold=best_t)
        tuning_results.append(["Ensemble", best_t] + tuned_metrics)
    
    tuning_headers = ["Model", "Best Thresh", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print(tabulate(tuning_results, headers=tuning_headers, floatfmt=".4f", tablefmt="grid"))

    # === Youden's Index ===
    print("\n=== Optimal Threshold (Youden's J) ===")
    youden_results = []
    for name, preds in [("ESM-MLP", seq_preds), ("GAT", graph_preds)]:
        best_t, best_j = find_optimal_threshold(labels, preds, method="youden")
        tuned_metrics = calc_metrics(labels, preds, threshold=best_t)
        youden_results.append([name, best_t, best_j] + tuned_metrics)
    
    if ens_preds is not None:
        best_t, best_j = find_optimal_threshold(labels, ens_preds, method="youden")
        tuned_metrics = calc_metrics(labels, ens_preds, threshold=best_t)
        youden_results.append(["Ensemble", best_t, best_j] + tuned_metrics)
    
    youden_headers = ["Model", "Best Thresh", "Youden J", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print(tabulate(youden_results, headers=youden_headers, floatfmt=".4f", tablefmt="grid"))

    # Generate SHAP summary plot
    if ensemble_model:
        print("\nGenerating SHAP Summary Plot...")
        explainer = PPIExplainer(str(ensemble_path))
        conf_seq = np.abs(seq_preds - 0.5)
        conf_gat = np.abs(graph_preds - 0.5)
        X_shap = np.column_stack((seq_preds, graph_preds, conf_seq, conf_gat))
        explainer.save_summary_plot(X_shap, output_path=str(PROJECT_ROOT / "data" / "processed" / "plots" / "shap_summary.png"))

if __name__ == "__main__":
    evaluate_models()
