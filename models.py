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
    
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphFusionNet_1(torch.nn.Module):
    def __init__(self, graph_conv_layer, in1, in2, hidden, out):
        super().__init__()
        # Two GCNs (one for each static data and the other for dynamic data)
        self.gcn1_1 = graph_conv_layer(in1, hidden)
        self.gcn1_2 = graph_conv_layer(in2, hidden)

        # Fusion weights (learnable)
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))

        self.gcn2_1 = graph_conv_layer(hidden, out)

    def forward(self, x1, x2, edge_index, return_embedding = False):
        # First-level message passing
        h1 = F.relu(self.gcn1_1(x1, edge_index))
        h2 = F.relu(self.gcn1_2(x2, edge_index))

        # Fusion (weighted sum)
        h = self.alpha * h1 + (1 - self.alpha) * h2

        # Second-level message passing
        out = self.gcn2_1(h, edge_index)

        return h if return_embedding else out


