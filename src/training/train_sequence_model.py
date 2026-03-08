import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
from sklearn.metrics import f1_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.sequence_model import SequencePPIModel
from src.utils.dataset import PPIDataset
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

def train(epochs: int = 50, batch_size: int = 64, lr: float = 5e-4, embedding_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Sequence Model on {device}...")
    
    if embedding_path and os.path.exists(embedding_path):
        embeddings = torch.load(embedding_path, weights_only=False)
    else:
        print("No embedding file provided/found. Cannot proceed without embeddings.")
        return

    # Datasets
    train_dataset = PPIDataset(PROCESSED_DATA_DIR / "train.csv", embeddings)
    val_dataset = PPIDataset(PROCESSED_DATA_DIR / "val.csv", embeddings)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    # Calculate class weights for imbalanced data
    train_labels = train_dataset.df["label"].values
    pos_count = train_labels.sum()
    neg_count = len(train_labels) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)
    print(f"Class balance — Positive: {pos_count}, Negative: {neg_count}, pos_weight: {pos_weight.item():.3f}")
    
    # Model
    sample_emb = next(iter(embeddings.values()))
    input_dim = sample_emb.shape[-1] if sample_emb.dim() > 1 else sample_emb.shape[0]
    
    model = SequencePPIModel(input_dim=input_dim, hidden_dim=256, dropout=0.3).to(device)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # BCEWithLogitsLoss (model now outputs raw logits)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Cosine Annealing scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for emb1, emb2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(emb1, emb2)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation with F1 tracking
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for emb1, emb2, labels in val_loader:
                emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
                outputs = model(emb1, emb2)
                probs = torch.sigmoid(outputs).squeeze()
                predicted = (probs > 0.5).float()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_acc = (all_preds == all_labels).mean()
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        avg_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}")
        
        # Save best by F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), PROJECT_ROOT / "models" / "sequence_model_best.pth")
            print(f"  ✓ Saved best model (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no F1 improvement for {patience} epochs)")
                break
    
    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to dictionary of protein embeddings (.pt)")
    args = parser.parse_args()
    
    train(args.epochs, args.batch_size, args.lr, args.embedding_path)
