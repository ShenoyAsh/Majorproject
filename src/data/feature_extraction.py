import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import sys
import os
import gzip
from Bio import SeqIO

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.paths import PROCESSED_DATA_DIR

class ESMFeatureExtractor:
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", device: str = "cpu"):
        self.device = device
        print(f"Loading ESM-2 model: {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def get_embeddings(self, sequences: Dict[str, str], batch_size: int = 16) -> Dict[str, torch.Tensor]:
        """
        Generates embeddings for a dictionary of sequences.
        Returns a dictionary mapping ProteinID -> Embedding Tensor.
        """
        embeddings = {}
        ids = list(sequences.keys())
        seqs = list(sequences.values())
        
        for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting features"):
            batch_ids = ids[i:i+batch_size]
            batch_seqs = seqs[i:i+batch_size]
            
            inputs = self.tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            for j, pid in enumerate(batch_ids):
                seq_len = len(batch_seqs[j])
                # Cap sequence length to what was processed minus EOS (1024 max)
                actual_len = min(seq_len, 1024 - 2)
                # Take CLS token (index 0) + all amino acid tokens (indexes 1 to actual_len)
                emb = outputs.last_hidden_state[j, 0:actual_len+1, :]
                embeddings[pid] = emb.cpu()
                
        return embeddings

if __name__ == "__main__":
    import numpy as np
    
    # 1. Load all proteins
    print("Loading datasets...")
    proteins = set()
    for split in ["train", "val", "test"]:
        path = PROCESSED_DATA_DIR / f"{split}.csv"
        if path.exists():
            df = pd.read_csv(path)
            proteins.update(df["protein1"].unique())
            proteins.update(df["protein2"].unique())
    
    print(f"Total unique proteins: {len(proteins)}")
    
    # 2. Get Sequences
    from src.utils.paths import STRING_SEQUENCES_FILE
    import gzip
    
    print(f"Loading sequences from {STRING_SEQUENCES_FILE}...")
    sequences = {}
    
    if not STRING_SEQUENCES_FILE.exists():
        print(f"Error: Sequences file not found at {STRING_SEQUENCES_FILE}")
        # Fallback to dummy? No, better to fail or warn.
        print("Falling back to dummy sequences for demonstration (NOT RECOMMENDED for production).")
        for p in proteins:
            length = np.random.randint(50, 100)
            aa = "ACDEFGHIKLMNPQRSTVWY"
            seq = "".join(np.random.choice(list(aa), length))
            sequences[p] = seq
    else:
        # STRING Fasta header example: >9606.ENSP00000000233|...
        # We need to strip "9606." to match our IDs which are "ENSP..."
        with gzip.open(STRING_SEQUENCES_FILE, "rt") as handle:
             for record in SeqIO.parse(handle, "fasta"):
                 # ID cleaning
                 full_id = record.id 
                 # Expecting 9606.ENSP...
                 if full_id.startswith("9606."):
                     protein_id = full_id[5:]
                 else:
                     protein_id = full_id
                     
                 # Check if this protein is in our dataset
                 if protein_id in proteins:
                     sequences[protein_id] = str(record.seq)

    print(f"Loaded {len(sequences)} sequences for {len(proteins)} required proteins.")
    
    # Check coverage
    missing = len(proteins) - len(sequences)
    if missing > 0:
        print(f"Warning: {missing} proteins missing sequences. Filling with dummy.")
        # Fill missing with dummy to prevent crash
        for p in proteins:
            if p not in sequences:
                length = np.random.randint(50, 100)
                aa = "ACDEFGHIKLMNPQRSTVWY"
                seq = "".join(np.random.choice(list(aa), length))
                sequences[p] = seq
        
    # 3. Extract Embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ESMFeatureExtractor(device=device)
    # Batch size might need adjustment for long sequences or GPU memory
    embeddings = extractor.get_embeddings(sequences, batch_size=4) 
    
    # 4. Save
    out_path = PROCESSED_DATA_DIR / "embeddings.pt"
    torch.save(embeddings, out_path)
    print(f"Embeddings saved to {out_path}")
