import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, heads: int = 8, 
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers with residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        
        # Layer 1: in_channels → hidden_channels * heads
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        self.skip_projs.append(nn.Linear(in_channels, hidden_channels * heads) 
                               if in_channels != hidden_channels * heads else nn.Identity())
        
        # Middle layers: hidden_channels * heads → hidden_channels * heads
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
            self.skip_projs.append(nn.Identity())  # Same dim, identity skip
        
        # Final layer: hidden_channels * heads → hidden_channels (concat=False)
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_channels))
        self.skip_projs.append(nn.Linear(hidden_channels * heads, hidden_channels))
        
        # Link Prediction Classifier (enhanced)
        # Input: h_u, h_v, |h_u - h_v|, h_u * h_v, cosine_sim → hidden_channels * 4 + 1
        classifier_in = hidden_channels * 4 + 1
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_channels * 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
            # No Sigmoid — use BCEWithLogitsLoss during training
        )

    def encode(self, x, edge_index):
        for i in range(self.num_layers):
            residual = self.skip_projs[i](x)
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Add residual for all layers
            x = x + residual
        return x

    def forward(self, x, edge_index, edge_label_index):
        """
        Args:
            x: Node features (batch_nodes, in_channels)
            edge_index: Graph connectivity (2, num_edges)
            edge_label_index: Pairs to predict (2, num_pairs)
        Returns:
            Raw logits (no sigmoid). Apply sigmoid yourself during inference.
        """
        z = self.encode(x, edge_index)
        
        src, dst = edge_label_index
        h_u = z[src]
        h_v = z[dst]
        
        abs_diff = torch.abs(h_u - h_v)
        hadamard = h_u * h_v
        cosine_sim = F.cosine_similarity(h_u, h_v, dim=1).unsqueeze(1)
        
        emb_pair = torch.cat([h_u, h_v, abs_diff, hadamard, cosine_sim], dim=1)
        
        return self.classifier(emb_pair)
