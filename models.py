import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphConv, SuperGATConv, GATConv, GCNConv
from torch import nn

class EdgeFeatureAttentionGCN(nn.Module):
    """
    A Graph Convolutional Network (GCN) layer with learnable, edge-specific feature-wise attention.

    This layer extends standard GCNs by dynamically computing an attention vector for each edge,
    allowing the model to weigh neighbor contributions differently across feature dimensions.
    The attention coefficients are produced by an MLP that takes concatenated transformed features
    of source and target nodes as input, enabling adaptive and edge-dependent message passing.

    Args:
        in_channels (int): Number of input features per node.
        out_channels (int): Number of output features per node.
        hidden_att (int): Hidden dimension for the attention MLP.
        use_bias (bool): Whether to include a learnable bias term.

    Returns:
        Tensor: Node embeddings of shape [num_nodes, out_channels].
    """
    def __init__(self, in_channels, out_channels, hidden_att=16, use_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear feature transformation
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        # MLP to compute edge-wise attention per feature
        self.att_mlp = nn.Sequential(
            nn.Linear(2*out_channels, hidden_att),
            nn.ReLU(),
            nn.Linear(hidden_att, out_channels),
            nn.Sigmoid()  # attention between 0 and 1
        )

        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        for layer in self.att_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        # Step 1: Linear transform
        x_transformed = x @ self.weight  # [N, F_out]

        # Step 2: Add self-loops
        self_loop_edges = torch.arange(num_nodes, device=x.device)
        self_loop_edges = self_loop_edges.unsqueeze(0).repeat(2,1)
        edge_index = torch.cat([edge_index, self_loop_edges], dim=1)

        row, col = edge_index  # row: target, col: source

        # Step 3: Compute edge attention dynamically
        edge_features = torch.cat([x_transformed[row], x_transformed[col]], dim=1)
        alpha = self.att_mlp(edge_features)  # [num_edges, F_out]

        # Step 4: Degree normalization (optional)
        deg = torch.bincount(row, minlength=num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = (deg_inv_sqrt[row] * deg_inv_sqrt[col]).unsqueeze(-1)

        # Step 5: Message passing
        messages = x_transformed[col] * alpha * norm  # [num_edges, F_out]
        out = torch.zeros_like(x_transformed)
        out = out.index_add(0, row, messages)

        # Step 6: Bias
        if self.bias is not None:
            out += self.bias

        return out

class FastGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(FastGraphConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges] COO format
        """
        num_nodes = x.size(0)

        # Step 1: Add self-loops
        self_loop_edges = torch.arange(num_nodes, device=x.device)
        self_loop_edges = self_loop_edges.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loop_edges], dim=1)

        row, col = edge_index

        # Step 2: Linear transformation
        x = x @ self.weight

        # Step 3: Compute degree normalization
        deg = torch.bincount(row, minlength=num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Step 4: Message passing with scatter_add
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = torch.zeros_like(x)
        out = out.index_add(0, row, x[col] * norm.unsqueeze(-1))

        # Step 5: Add bias
        if self.bias is not None:
            out += self.bias

        return out

class FeatureWiseGraphConv(nn.Module):
    """
    A feature-wise Graph Convolution layer that applies a learnable, 
    feature-dependent attention weighting during message passing.

    This layer performs graph convolution by linearly transforming node features 
    and aggregating information from neighboring nodes, with each output feature 
    dimension having its own learnable attention coefficient (`alpha`). 
    The operation follows a normalized message-passing scheme with self-loops.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    out_channels : int
        Dimensionality of output node features.
    use_bias : bool, optional (default=True)
        If True, adds a learnable bias term to the output.

    Inputs
    ------
    x : torch.Tensor, shape [num_nodes, in_channels]
        Node feature matrix.
    edge_index : torch.LongTensor, shape [2, num_edges]
        Graph connectivity in COO format, where each column represents an edge (source, target).

    Returns
    -------
    out : torch.Tensor, shape [num_nodes, out_channels]
        Updated node feature representations after message passing.

    Notes
    -----
    - Adds self-loops to the graph to include each node's own features during aggregation.
    - Uses symmetric normalization (similar to GCN): 
      `norm_ij = (deg_i * deg_j)^(-1/2)` for each edge (i, j).
    - The learnable vector `alpha` scales each output feature dimension independently, 
      introducing a form of **feature-wise attention**.
    """
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(FeatureWiseGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear transformation
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        # Attention vector per feature (scalar per edge)
        # We'll initialize learnable alpha per feature
        self.alpha = nn.Parameter(torch.Tensor(out_channels))

        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.ones_(self.alpha)  # start with all ones
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges] COO format
        """
        num_nodes = x.size(0)

        # Add self-loops
        self_loop_edges = torch.arange(num_nodes, device=x.device)
        self_loop_edges = self_loop_edges.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loop_edges], dim=1)

        row, col = edge_index

        # Linear transform
        x = x @ self.weight  # [N, out_channels]

        # Compute degree normalization
        deg = torch.bincount(row, minlength=num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Message passing with feature-wise attention
        messages = x[col] * self.alpha  # element-wise per feature
        messages = messages * norm.unsqueeze(-1)

        out = torch.zeros_like(x)
        out = out.index_add(0, row, messages)

        # Add bias
        if self.bias is not None:
            out += self.bias

        return out
# ==========================================================
# Define 2-Layer GCN Model
# ==========================================================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.7):
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
    
class ImprovedGCN_2Layers(torch.nn.Module):
    def __init__(self, graph_conv_layer,in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv1 = graph_conv_layer(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = graph_conv_layer(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, return_embedding=False):
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x_emb = x1.clone()
        x_out = self.conv2(x1, edge_index)
        return x_emb if return_embedding else x_out



class ImprovedGCN_3Layers(torch.nn.Module):
    def __init__(self, graph_conv_layer, in_channels, hidden_channel_1=32, hidden_channel_2 = 64,  out_channels=2, dropout=0.1):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(in_channels)  # input normalization

        self.conv1 = graph_conv_layer(in_channels, hidden_channel_1)
        self.ln1 = nn.LayerNorm(hidden_channel_1)
        self.conv2 = graph_conv_layer(hidden_channel_1, hidden_channel_2)
        self.ln2 = nn.LayerNorm(hidden_channel_2)
        self.conv3 = graph_conv_layer(hidden_channel_2, out_channels)
        # Projection layer for residuals when dimensions differ
        self.res_proj = nn.Linear(hidden_channel_1, hidden_channel_2)

        self.dropout = dropout

    def forward(self, x, edge_index, return_embedding=False):
        # Normalize input features
        x = self.norm_input(x)

        # Layer 1
        h1 = self.conv1(x, edge_index)
        h1 = self.ln1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2 with residual connection
        h2 = self.conv2(h1, edge_index)
        h2 = self.ln2(h2)
        # Project h1 → h2 size if needed
        if h1.shape[1] != h2.shape[1]:
            h1 = self.res_proj(h1)
        h2 = F.relu(h2 + h1)  # residual connection
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        if return_embedding:
            return h2

        # Output layer
        out = self.conv3(h2, edge_index)
        return out
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedGCN_4Layers(nn.Module):
    def __init__(self, graph_conv_layer, in_channels,
                 hidden_channel_1=32, hidden_channel_2=64, hidden_channel_3=128, out_channels=2, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        # Normalization on input
        self.norm_input = nn.BatchNorm1d(in_channels)

        # Graph convolution layers
        self.conv1 = graph_conv_layer(in_channels, hidden_channel_1)
        self.ln1 = nn.LayerNorm(hidden_channel_1)

        self.conv2 = graph_conv_layer(hidden_channel_1, hidden_channel_2)
        self.ln2 = nn.LayerNorm(hidden_channel_2)

        self.conv3 = graph_conv_layer(hidden_channel_2, hidden_channel_3)
        self.ln3 = nn.LayerNorm(hidden_channel_3)

        self.conv4 = graph_conv_layer(hidden_channel_3, out_channels)

        # Projection layers for residuals if needed
        self.res_proj_1to2 = nn.Linear(hidden_channel_1, hidden_channel_2)
        self.res_proj_2to3 = nn.Linear(hidden_channel_2, hidden_channel_3)

    def forward(self, x, edge_index, return_embedding=False):
        # === Layer 1 ===
        x = self.norm_input(x)
        h1 = F.relu(self.ln1(self.conv1(x, edge_index)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # === Layer 2 ===
        h2 = F.relu(self.ln2(self.conv2(h1, edge_index)))
        if h1.shape[1] != h2.shape[1]:
            h1_proj = self.res_proj_1to2(h1)
        else:
            h1_proj = h1
        h2 = F.relu(h2 + h1_proj)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        # === Layer 3 ===
        h3 = F.relu(self.ln3(self.conv3(h2, edge_index)))
        if h2.shape[1] != h3.shape[1]:
            h2_proj = self.res_proj_2to3(h2)
        else:
            h2_proj = h2
        h3 = F.relu(h3 + h2_proj)
        h3 = F.dropout(h3, p=self.dropout, training=self.training)

        # === Output Layer ===
        if return_embedding:
            return h3

        out = self.conv4(h3, edge_index)
        return out

import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch import nn

class APPNPNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.05, K=20, alpha=0.2):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = APPNP(K, alpha, dropout=dropout)  # K=number of propagation steps

    def forward(self, x, edge_index, return_embedding=False):
        x = F.dropout(x, p=0.05, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.05, training=self.training)
        x_emb = x.clone()
        x = self.lin2(x)
        x = self.prop(x, edge_index)

        return x_emb if return_embedding else x
    
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, JumpingKnowledge, BatchNorm

class AdvancedGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=4, dropout=0.5):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        self.bns.append(BatchNorm(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.bns.append(BatchNorm(hidden_channels * heads))

        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))

        # Jumping Knowledge: aggregate all layer outputs
        self.jump = JumpingKnowledge(mode='cat')  # can be 'max' or 'lstm' too

        # Linear projection for JK concatenation
        self.lin = torch.nn.Linear(out_channels + (num_layers - 1) * hidden_channels * heads, out_channels)

    def forward(self, x, edge_index, return_embedding=False):
        xs = []  # store layer outputs for JK aggregation

        for i, conv in enumerate(self.convs[:-1]):
            residual = x  # save for skip connection
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)  # ELU smoother than ReLU for attention
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual if residual.shape == x.shape else x  # residual connection
            xs.append(x)

        # Final layer (no BN)
        x_out = self.convs[-1](x, edge_index)
        xs.append(x_out)

        # Jumping Knowledge fusion
        x = self.jump(xs)

        # Projection
        x = self.lin(x)

        return torch.cat(xs, dim=-1) if return_embedding else x



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


