import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


# ==========================================================
# Define 2-Layer GCN Model
# ==========================================================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, return_embedding=False):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Save embedding before classification
        x_emb = x.clone()

        # Layer 2: classification logits
        x_out = self.conv2(x, edge_index)

        return x_emb if return_embedding else x_out

