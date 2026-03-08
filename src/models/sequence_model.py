import torch
import torch.nn as nn

class SequencePPIModel(nn.Module):
    def __init__(self, input_dim: int = 320, hidden_dim: int = 256, dropout: float = 0.3):
        """
        Enhanced MLP model for PPI prediction using protein embeddings.
        Uses feature interaction: concat, abs_diff, hadamard → 4× input width.
        Outputs raw logits (no sigmoid) for use with BCEWithLogitsLoss.
        """
        super().__init__()
        
        # Input: [e1, e2, |e1-e2|, e1*e2] → 4 * input_dim
        combined_dim = input_dim * 4
        
        # Layer 1
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Layer 3
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Layer 4 (output)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.1)
        
        # Residual projection for skip connection (fc1 → fc2 have same dim)
        # fc1 and fc2 both output hidden_dim, so identity skip works

    def forward(self, emb1, emb2):
        """
        Args:
            emb1: Tensor of shape (batch, input_dim)
            emb2: Tensor of shape (batch, input_dim)
        Returns:
            Raw logits (no sigmoid). Apply sigmoid during inference.
        """
        # Feature interaction
        abs_diff = torch.abs(emb1 - emb2)
        hadamard = emb1 * emb2
        x = torch.cat([emb1, emb2, abs_diff, hadamard], dim=1)
        
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        # Layer 2 with residual
        residual = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x + residual  # skip connection
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.dropout(x)
        
        # Output (raw logits)
        x = self.fc4(x)
        return x
