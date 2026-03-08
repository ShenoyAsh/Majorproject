import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict

class PPIDataset(Dataset):
    def __init__(self, data_path: str, embeddings: Dict[str, torch.Tensor], return_type: str = "cls"):
        """
        Args:
            data_path: Path to the csv file (protein1, protein2, label).
            embeddings: Dictionary mapping ProteinID -> Embedding.
            return_type: "cls", "sequence", or "full". Determines the portion of the embedding to return.
        """
        self.df = pd.read_csv(data_path)
        self.embeddings = embeddings
        self.return_type = return_type
        
        # Filter out pairs where embeddings are missing
        # In a real scenario, we should handle this earlier or ensure completeness.
        valid_indices = []
        for idx, row in self.df.iterrows():
            if row["protein1"] in self.embeddings and row["protein2"] in self.embeddings:
                valid_indices.append(idx)
        
        if len(valid_indices) < len(self.df):
            print(f"Warning: Dropped {len(self.df) - len(valid_indices)} pairs due to missing embeddings.")
            self.df = self.df.loc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p1 = row["protein1"]
        p2 = row["protein2"]
        label = row["label"]
        
        emb1_full = self.embeddings[p1]
        emb2_full = self.embeddings[p2]
        
        # Handle backward compatibility: older embeddings might just be 1D
        if emb1_full.dim() == 1:
            emb1 = emb1_full
            emb2 = emb2_full
        else:
            if self.return_type == "cls":
                emb1 = emb1_full[0]
                emb2 = emb2_full[0]
            elif self.return_type == "sequence":
                emb1 = emb1_full[1:] # Sequence without CLS
                emb2 = emb2_full[1:]
            else: # "full"
                emb1 = emb1_full
                emb2 = emb2_full
        
        return emb1, emb2, torch.tensor(label, dtype=torch.float32)

def ppi_collate_fn(batch):
    # If 1D (CLS), standard stack
    if batch[0][0].dim() == 1:
        emb1_batch = torch.stack([item[0] for item in batch])
        emb2_batch = torch.stack([item[1] for item in batch])
        labels_batch = torch.stack([item[2] for item in batch])
        return emb1_batch, emb2_batch, labels_batch
    
    # For 2D sequence, dynamic padding
    from torch.nn.utils.rnn import pad_sequence
    emb1_list = [item[0] for item in batch]
    emb2_list = [item[1] for item in batch]
    labels_batch = torch.stack([item[2] for item in batch])
    
    emb1_padded = pad_sequence(emb1_list, batch_first=True, padding_value=0.0)
    emb2_padded = pad_sequence(emb2_list, batch_first=True, padding_value=0.0)
    
    # Create padding masks for attention (True for padding)
    mask1 = torch.zeros(len(emb1_list), emb1_padded.size(1), dtype=torch.bool)
    mask2 = torch.zeros(len(emb2_list), emb2_padded.size(1), dtype=torch.bool)
    
    for i, seq in enumerate(emb1_list):
        mask1[i, len(seq):] = True
    for i, seq in enumerate(emb2_list):
        mask2[i, len(seq):] = True
        
    return emb1_padded, mask1, emb2_padded, mask2, labels_batch
