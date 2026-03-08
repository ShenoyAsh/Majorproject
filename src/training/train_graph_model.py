import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd
import random
from sklearn.metrics import f1_score
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.graph_model import GATLinkPredictor
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def train(epochs: int = 200, lr: float = 0.001, graph_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Graph Model on {device}...")
    
    if graph_path and os.path.exists(graph_path):
        data = torch.load(graph_path, weights_only=False)
        data = data.to(device)
    else:
        print("Graph file not found.")
        return

    # Load Splits
    print("Loading datasets for link prediction...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    
    # Load node mapping
    mapping_path = str(graph_path).replace(".pt", "_mapping.pt")
    if os.path.exists(mapping_path):
        node_mapping = torch.load(mapping_path, weights_only=False)
    else:
        print("Node mapping not found. Cannot align CSVs to Graph.")
        return

    # Helper to get edge indices from DF
    def get_edge_label_index(df):
        src = []
        dst = []
        labels = []
        for _, row in df.iterrows():
            if row["protein1"] in node_mapping and row["protein2"] in node_mapping:
                src.append(node_mapping[row["protein1"]])
                dst.append(node_mapping[row["protein2"]])
                labels.append(row["label"])
        return torch.tensor([src, dst], dtype=torch.long), torch.tensor(labels, dtype=torch.float32)

    train_edge_label_index, train_labels = get_edge_label_index(train_df)
    val_edge_label_index, val_labels = get_edge_label_index(val_df)
    
    train_edge_label_index, train_labels = train_edge_label_index.to(device), train_labels.to(device)
    val_edge_label_index, val_labels = val_edge_label_index.to(device), val_labels.to(device)

    # Calculate class info & apply recall-boosting pos_weight
    pos_count = train_labels.sum().item()
    neg_count = len(train_labels) - pos_count
    # Override pos_weight to 1.5 — penalizes false negatives harder to improve recall
    # Even on balanced data, GATs tend to be overly conservative
    pos_weight = torch.tensor([1.5], dtype=torch.float32).to(device)
    print(f"Class balance — Positive: {int(pos_count)}, Negative: {int(neg_count)}, pos_weight: {pos_weight.item():.3f} (recall-boosted)")

    # Separate positive and negative indices for negative subsampling
    pos_mask = train_labels == 1
    neg_mask = train_labels == 0
    pos_indices = torch.where(pos_mask)[0]
    neg_indices = torch.where(neg_mask)[0]
    print(f"Negative subsampling enabled: 70% of negatives used per epoch")

    # Model — enhanced architecture
    model = GATLinkPredictor(
        in_channels=data.x.shape[1], 
        hidden_channels=128, 
        heads=8, 
        num_layers=3, 
        dropout=0.3
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # BCEWithLogitsLoss with positive weight to fix low recall
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Cosine Annealing scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_val_f1 = 0.0
    patience = 15
    patience_counter = 0

    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Negative subsampling — use all positives + random 70% of negatives
        # This forces the model to lean toward predicting positive → higher recall
        neg_sample_size = int(len(neg_indices) * 0.7)
        sampled_neg = neg_indices[torch.randperm(len(neg_indices))[:neg_sample_size]]
        epoch_indices = torch.cat([pos_indices, sampled_neg])
        epoch_indices = epoch_indices[torch.randperm(len(epoch_indices))]  # shuffle
        
        epoch_edge_label_index = train_edge_label_index[:, epoch_indices]
        epoch_labels = train_labels[epoch_indices]
        
        # Forward pass — model returns raw logits
        outputs = model(data.x, data.edge_index, epoch_edge_label_index)
        loss = criterion(outputs.squeeze(), epoch_labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(data.x, data.edge_index, val_edge_label_index)
            val_probs = torch.sigmoid(val_outputs.squeeze())
            val_preds = (val_probs > 0.5).float()
            
            val_acc = (val_preds == val_labels).sum().item() / val_labels.size(0)
            
            # Calculate F1
            val_preds_np = val_preds.cpu().numpy()
            val_labels_np = val_labels.cpu().numpy()
            val_f1 = f1_score(val_labels_np, val_preds_np, zero_division=0)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}")
        
        # Save best model by F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), PROJECT_ROOT / "models" / "graph_model_best.pth")
            print(f"  ✓ Saved best model (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no F1 improvement for {patience} epochs)")
                break
    
    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--graph_path", type=str, required=True, help="Path to PyG Data object (.pt)")
    args = parser.parse_args()
    
    train(args.epochs, args.lr, args.graph_path)
