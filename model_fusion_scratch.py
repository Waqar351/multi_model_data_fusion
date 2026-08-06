import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import NeighborSampler  # optional for large graphs

# -------------------------
# Utility / simple MLP
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(128,), activate_last=False):
        super().__init__()
        layers = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU(inplace=True))
            cur = h
        layers.append(nn.Linear(cur, out_dim))
        if activate_last:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------
# Custom Message Passing
# -------------------------
class CustomMP(MessagePassing):
    def __init__(self, in_channels, out_channels, use_edge_weights=False):
        # using 'add' or 'mean' aggregation; mean typically stabilizes
        super().__init__(aggr='mean')
        self.phi = nn.Linear(in_channels, out_channels)   # transform neighbor features
        self.psi = nn.Linear(out_channels, out_channels)  # post-aggregation transform
        self.use_edge_weights = use_edge_weights
        # edge weights will be registered externally as a parameter on the model

    def forward(self, x, edge_index, edge_weight=None):
        # x: [N, in_channels]
        # edge_index: [2, E]
        # edge_weight: [E] or None
        # ensure self loops so that each node can see itself (optional)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if edge_weight is not None:
            # if self-loops were added, append ones for them
            pad = x.new_ones(x.size(0))
            edge_weight = torch.cat([edge_weight, pad], dim=0)
        # propagate automatically calls message + aggregate + update
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.psi(out)
        return out

    def message(self, x_j, edge_weight=None):
        # x_j: neighbor features (source node features)
        m = self.phi(x_j)
        if edge_weight is not None:
            # edge_weight shape matches number of messages
            return m * edge_weight.view(-1, 1)
        return m

    def update(self, aggr_out):
        # aggr_out is the aggregated message per node
        return F.relu(aggr_out)

# -------------------------
# Fusion gate
# -------------------------
class FusionGate(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        # gate network produces scalar alpha in (0,1) per node
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, z_static, z_dynamic):
        # z_static/z_dynamic: [N, dim]
        z = torch.cat([z_static, z_dynamic], dim=1)            # [N, 2*dim]
        alpha = torch.sigmoid(self.net(z)).view(-1, 1)         # [N, 1]
        fused = alpha * z_static + (1.0 - alpha) * z_dynamic  # [N, dim]
        return fused, alpha

# -------------------------
# Dual-input GNN
# -------------------------
class DualGNN(nn.Module):
    """
    Dual-stream Graph Neural Network that integrates static and dynamic node features through gated fusion
    and performs adaptive message passing with optional learnable edge weights.

    The model first encodes static and dynamic node attributes into a shared latent space using MLP encoders.
    A FusionGate adaptively merges these two representations per node. The fused embeddings are then refined
    through multiple message-passing layers (CustomMP), optionally modulated by learnable edge weights to
    capture varying relational strengths. The final node embeddings are passed through a classifier for prediction.

    Args:
        in_static (int): Dimension of static node features.
        in_dynamic (int): Dimension of dynamic node features.
        hidden_dim (int): Dimensionality of encoder outputs and fusion space.
        message_dim (int): Hidden size used in message-passing layers.
        num_classes (int): Number of output classes.
        use_edge_weights (bool): Whether to use learnable edge weights for message modulation.

    Returns:
        Tuple[Tensor, Tensor]: 
            - Node-level class predictions of shape [N, num_classes].
            - Fusion coefficients (alpha) indicating the fusion ratio between static and dynamic features.
    """
    def __init__(self,
                 in_static,
                 in_dynamic,
                 hidden_dim=64,
                 message_dim=64,
                 num_classes=2,
                 use_edge_weights=False):
        """
        in_static, in_dynamic: dims of raw static and dynamic features
        hidden_dim: encoder output size for each stream
        message_dim: hidden size used inside message passing
        """
        super().__init__()
        # encoders -> produce same dimension so we can fuse
        self.static_enc = MLP(in_static, hidden_dim, hidden_dims=(hidden_dim,))
        self.dynamic_enc = MLP(in_dynamic, hidden_dim, hidden_dims=(hidden_dim,))

        # fusion
        self.fusion = FusionGate(hidden_dim)

        # message passing layers (two layers example)
        self.mp1 = CustomMP(hidden_dim, message_dim, use_edge_weights=use_edge_weights)
        self.mp2 = CustomMP(message_dim, message_dim, use_edge_weights=use_edge_weights)

        # final classifier head
        self.classifier = nn.Sequential(
            nn.Linear(message_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, num_classes)
        )

        self.use_edge_weights = use_edge_weights
        self._edge_weight = None  # registered at runtime if requested

    def register_edge_weights(self, num_edges, device='cpu', init_value=1.0):
        """Call this once with number of edges (E) BEFORE training if you want learnable edge weights.
           Note: if you will call add_self_loops inside MP, you should ensure sizing accordingly.
        """
        w = torch.full((num_edges,), float(init_value), dtype=torch.float32, device=device)
        self._edge_weight = nn.Parameter(w)
        # register as parameter so optimizer updates it
        self.register_parameter("edge_weight_param", self._edge_weight)

    def forward(self, x_static, x_dynamic, edge_index, return_embedding=False, edge_index_original_size=None):
        """
        x_static: [N, in_static]
        x_dynamic: [N, in_dynamic]
        edge_index: PyG edge_index (2, E) -- this should be the edge_index WITHOUT self loops
        edge_index_original_size: if edge_weight param exists, its length must match E (without added self loops)
        """
        # 1) encode
        zs = self.static_enc(x_static)    # [N, hidden_dim]
        zd = self.dynamic_enc(x_dynamic)  # [N, hidden_dim]

        # 2) fuse with gate
        z, alpha = self.fusion(zs, zd)    # z: [N, hidden_dim], alpha: [N,1]

        # 3) optionally prepare edge weights
        edge_weight = None
        if self.use_edge_weights:
            if getattr(self, "_edge_weight", None) is None:
                raise RuntimeError("Edge weights not registered. Call model.register_edge_weights(E) first.")
            # use the param as edge weights for the given edge_index (no self loops yet)
            edge_weight = self._edge_weight
            # NOTE: CustomMP will append ones for self-loops internally

        # 4) message passing stack
        h = self.mp1(z, edge_index, edge_weight=edge_weight)   # [N, message_dim]
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.mp2(h, edge_index, edge_weight=edge_weight)   # [N, message_dim]

        if return_embedding:
            return h  # latent embedding

        # 5) classifier
        out = self.classifier(h)   # [N, num_classes]
        return out, alpha

###__________________________________________________________________________________________________
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# --- Fusion Gate ---
class FusionGate(nn.Module):
    # It concatenates static and dynamic features and then pass to the linear hidden and activation and linear layers
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, z_static, z_dynamic):
        z = torch.cat([z_static, z_dynamic], dim=1)      # [N, 2*dim]
        alpha = torch.sigmoid(self.net(z)).view(-1, 1)  # [N, 1]
        fused = alpha * z_static + (1.0 - alpha) * z_dynamic
        return fused, alpha

# --- GraphConv with independent input sizes but shared weight logic ---
class SharedGraphConv(MessagePassing):
    def __init__(self, in_static, in_dynamic, out_channels):
        super().__init__(aggr='add')
        # Separate projections for static and dynamic inputs
        self.W_static = nn.Parameter(torch.randn(in_static, out_channels))
        self.W_dynamic = nn.Parameter(torch.randn(in_dynamic, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x_static, x_dynamic, edge_index):
        # Project both to the same output dimension
        x_static_proj = torch.matmul(x_static, self.W_static)
        x_dynamic_proj = torch.matmul(x_dynamic, self.W_dynamic)

        # Combine both graphs (optional: could also propagate separately)
        x_static_prop = self.propagate(edge_index, x=x_static_proj)
        x_dynamic_prop = self.propagate(edge_index, x=x_dynamic_proj)

        # Return both embeddings
        return x_static_prop + self.bias, x_dynamic_prop + self.bias

    def message(self, x_j):
        return x_j
   

# --- Full model ---
class StaticDynamicGCN(nn.Module):
    """
    StaticDynamicGCN performs joint graph learning from two distinct feature modalities (static and dynamic)
    using a shared graph convolutional framework followed by a gated feature fusion mechanism.

    The model first applies a SharedGraphConv layer that independently projects static and dynamic node features 
    into a common latent space while propagating structural information through the graph. 
    The resulting feature embeddings are adaptively fused by a FusionGate, which learns a per-node weighting between 
    static and dynamic representations. The fused node embeddings are then passed through a linear output layer for 
    downstream prediction (e.g., node classification).

    Args:
        in_static (int): Dimensionality of static input features.
        in_dynamic (int): Dimensionality of dynamic input features.
        hidden_dim (int): Dimension of the latent representation after graph convolution.
        out_dim (int): Dimension of the final output (e.g., number of classes).

    Returns:
        Tuple[Tensor, Tensor]:
            - Model output logits of shape [N, out_dim].
            - Fusion coefficients (alpha) of shape [N, 1], indicating the contribution of static vs. dynamic features.
    """
    def __init__(self, in_static, in_dynamic, hidden_dim, out_dim):
        super().__init__()
        self.gconv = SharedGraphConv(in_static, in_dynamic, hidden_dim)
        self.fusion_gate = FusionGate(hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_static, x_dynamic, edge_index, return_embedding = False):
        # Apply shared graph conv with separate projections
        z_static, z_dynamic = self.gconv(x_static, x_dynamic, edge_index)
        
        # Fuse via gate
        z_fused, alpha = self.fusion_gate(z_static, z_dynamic)
        
        if return_embedding:
            return z_fused  # latent embedding
        # Output
        out = self.out(z_fused)
        return out, alpha

###_______________________________________________________________________________________

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class StaticDynamicGNN_nonato(MessagePassing):
    def __init__(self, in_static, in_dynamic, out_channels):
        super().__init__(aggr='add')  # sum aggregation
        # Node-level weights
        self.W_stat_self = nn.Parameter(torch.randn(in_static, out_channels))
        self.W_dyn_self = nn.Parameter(torch.randn(in_dynamic, out_channels))
        # Neighbor-level weights
        self.W_stat_neigh = nn.Parameter(torch.randn(in_static, out_channels))
        self.W_dyn_neigh = nn.Parameter(torch.randn(in_dynamic, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x_static, x_dynamic, edge_index, return_embedding=False):
        # Self-node contributions
        self_stat = torch.matmul(x_static, self.W_stat_self)
        self_dyn = torch.matmul(x_dynamic, self.W_dyn_self)
        # Neighbor contributions via propagate
        neigh_stat = self.propagate(edge_index, x=x_static, weight=self.W_stat_neigh)
        neigh_dyn = self.propagate(edge_index, x=x_dynamic, weight=self.W_dyn_neigh)
        
        # Track contributions at dataset level
        contrib_static = (self_stat + neigh_stat).abs().mean()
        contrib_dynamic = (self_dyn + neigh_dyn).abs().mean()
        # Combine
        out = self_stat + self_dyn + neigh_stat + neigh_dyn + self.bias
        return out, {'static_dataset': contrib_static.item(), 'dynamic_dataset': contrib_dynamic.item()}
    
    def message(self, x_j, weight):
        # Linear transform for neighbor features
        return torch.matmul(x_j, weight)
    
###____________________________________________________________________________________________________________

class StaticDynamicGNN_hidden(MessagePassing):
    def __init__(self, in_static, in_dynamic, hidden_dim, out_channels):
        super().__init__(aggr='add')
        # Hidden layer weights
        self.W_stat_self_h = nn.Parameter(torch.randn(in_static, hidden_dim))
        self.W_dyn_self_h = nn.Parameter(torch.randn(in_dynamic, hidden_dim))
        self.W_stat_neigh_h = nn.Parameter(torch.randn(in_static, hidden_dim))
        self.W_dyn_neigh_h = nn.Parameter(torch.randn(in_dynamic, hidden_dim))
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))
        # Output layer weights
        self.W_out = nn.Parameter(torch.randn(hidden_dim, out_channels))
        self.b_out = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x_static, x_dynamic, edge_index, return_embedding=False):
        # Hidden embeddings
        hidden_self = torch.matmul(x_static, self.W_stat_self_h) + torch.matmul(x_dynamic, self.W_dyn_self_h)
        hidden_neigh = self.propagate(edge_index, x=x_static, weight=self.W_stat_neigh_h) + \
                       self.propagate(edge_index, x=x_dynamic, weight=self.W_dyn_neigh_h)
        hidden = F.relu(hidden_self + hidden_neigh + self.b_h)
        
        # Track hidden contributions
        contrib_static = (torch.matmul(x_static, self.W_stat_self_h) + self.propagate(edge_index, x=x_static, weight=self.W_stat_neigh_h)).abs().mean()
        contrib_dynamic = (torch.matmul(x_dynamic, self.W_dyn_self_h) + self.propagate(edge_index, x=x_dynamic, weight=self.W_dyn_neigh_h)).abs().mean()
        
        # Output layer
        out = torch.matmul(hidden, self.W_out) + self.b_out
        
        if return_embedding:
            return hidden
        else:
            return out, {'static_dataset': contrib_static.item(),
                         'dynamic_dataset': contrib_dynamic.item()}
    
    def message(self, x_j, weight):
        return torch.matmul(x_j, weight)
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class StaticDynamicGNN_adv(MessagePassing):
    """
    StaticDynamicGNN_hidden
    -----------------------
    A Graph Neural Network layer that jointly processes static and dynamic node features.
    It integrates feature contributions from self and neighboring nodes using separate
    learnable transformations for static and dynamic modalities, followed by a shared
    hidden representation and output projection.

    Attributes
    ----------
    in_static : int
        Dimensionality of static node features.
    in_dynamic : int
        Dimensionality of dynamic node features.
    hidden_dim : int
        Dimensionality of the intermediate hidden representation.
    out_channels : int
        Number of output channels (task-dependent).
    """

    def __init__(self, in_static, in_dynamic, hidden_dim, out_channels, dropout=0.2):
        super().__init__(aggr='add')  # Aggregate messages by summation

        # --- Linear layers for hidden transformations ---
        self.lin_stat_self = nn.Linear(in_static, hidden_dim)
        self.lin_dyn_self = nn.Linear(in_dynamic, hidden_dim)
        self.lin_stat_neigh = nn.Linear(in_static, hidden_dim)
        self.lin_dyn_neigh = nn.Linear(in_dynamic, hidden_dim)

        # --- Output projection ---
        self.lin_out = nn.Linear(hidden_dim, out_channels)

        # --- Normalization and regularization ---
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

        # --- Activation ---
        self.act = nn.LeakyReLU(negative_slope=0.1)

        # --- Initialization ---
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all parameters using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_static, x_dynamic, edge_index, return_embedding=False):
        """
        Forward propagation of node features through the message-passing layer.

        Parameters
        ----------
        x_static : torch.Tensor
            Static node feature matrix of shape [N, in_static].
        x_dynamic : torch.Tensor
            Dynamic node feature matrix of shape [N, in_dynamic].
        edge_index : torch.LongTensor
            Graph connectivity in COO format with shape [2, E].
        return_embedding : bool, optional
            Whether to return hidden embeddings instead of the final output.

        Returns
        -------
        torch.Tensor or (torch.Tensor, dict)
            If return_embedding=True, returns the hidden representation tensor.
            Otherwise, returns (output tensor, contribution dictionary).
        """

        # --- Self and neighborhood transformations ---
        hidden_self = self.lin_stat_self(x_static) + self.lin_dyn_self(x_dynamic)
        hidden_neigh = self.propagate(edge_index, x_static=x_static, x_dynamic=x_dynamic)
        
        # --- Hidden activation ---
        hidden = self.act(hidden_self + hidden_neigh)
        hidden = self.batch_norm(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        # --- Contribution tracking ---
        contrib_static = self.lin_stat_self(x_static).abs().mean().item()
        contrib_dynamic = self.lin_dyn_self(x_dynamic).abs().mean().item()

        # --- Output layer ---
        out = self.lin_out(hidden)

        if return_embedding:
            return hidden, {
                'static_dataset': contrib_static,
                'dynamic_dataset': contrib_dynamic
            }
        else:
            return out, {
                'static_dataset': contrib_static,
                'dynamic_dataset': contrib_dynamic
            }

    def message(self, x_static_j, x_dynamic_j):
        """
        Message computation for each neighbor j.

        Parameters
        ----------
        x_static_j : torch.Tensor
            Static features of neighboring nodes.
        x_dynamic_j : torch.Tensor
            Dynamic features of neighboring nodes.

        Returns
        -------
        torch.Tensor
            Combined message embedding for each edge.
        """
        msg = self.lin_stat_neigh(x_static_j) + self.lin_dyn_neigh(x_dynamic_j)
        return msg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import SAGEConv, GraphConv


class StaticDynamicGNN_adv_2(MessagePassing):
    """
    StaticDynamicGNN_adv
    --------------------
    A Graph Neural Network layer that jointly processes static and dynamic node features.
    It integrates feature contributions from self and neighboring nodes using separate
    learnable transformations for static and dynamic modalities, followed by a shared
    hidden representation and output projection.

    Attributes
    ----------
    in_static : int
        Dimensionality of static node features.
    in_dynamic : int
        Dimensionality of dynamic node features.
    hidden_dim : int
        Dimensionality of the intermediate hidden representation.
    out_channels : int
        Number of output channels (task-dependent).
    """

    def __init__(self, in_static, in_dynamic, hidden_dim, out_channels, dropout=0.2):
        super().__init__(aggr='add')  # aggregate messages by summation

        # --- Linear transformations ---
        self.lin_stat_self = nn.Linear(in_static, hidden_dim)
        self.lin_dyn_self = nn.Linear(in_dynamic, hidden_dim)
        self.lin_stat_neigh = nn.Linear(in_static, hidden_dim)
        self.lin_dyn_neigh = nn.Linear(in_dynamic, hidden_dim)

        # --- Output projection ---
        self.lin_out = nn.Linear(hidden_dim, out_channels)

        # --- Normalization and regularization ---
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

        # --- Activation ---
        self.act = nn.LeakyReLU(negative_slope=0.1)

        # --- Initialize parameters ---
        self.reset_parameters()

    # ---------------- Initialization ---------------- #
    def reset_parameters(self):
        """Initialize all parameters using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------- Message Passing ---------------- #
    def message(self, x_static_j, x_dynamic_j):
        """
        Compute messages for neighboring nodes j.
        """
        msg = self.lin_stat_neigh(x_static_j) + self.lin_dyn_neigh(x_dynamic_j)
        return msg

    # ---------------- Forward Pass ---------------- #
    def forward(self, x_static, x_dynamic, edge_index, return_embedding=False):
        """
        Forward pass through the layer.

        Parameters
        ----------
        x_static : torch.Tensor
            Static node features of shape [N, in_static].
        x_dynamic : torch.Tensor
            Dynamic node features of shape [N, in_dynamic].
        edge_index : torch.LongTensor
            Graph connectivity in COO format [2, E].
        return_embedding : bool, optional
            Whether to return hidden embeddings instead of the final output.

        Returns
        -------
        torch.Tensor or (torch.Tensor, dict)
            Output tensor or (output tensor, contribution dictionary).
        """

        # --- Compute self and neighbor embeddings ---
        hidden_self = self.lin_stat_self(x_static) + self.lin_dyn_self(x_dynamic)
        hidden_neigh = self.propagate(edge_index, x_static=x_static, x_dynamic=x_dynamic)

        # --- Combine, normalize, activate ---
        hidden = self.act(hidden_self + hidden_neigh)
        hidden = self.batch_norm(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        # --- Compute output ---
        out = self.lin_out(hidden)

        # --- Compute modality contributions ---
        contrib_dict = self._compute_contributions(x_static, x_dynamic)

        if return_embedding:
            return hidden, contrib_dict
        else:
            return out, contrib_dict

    # ---------------- Contribution Tracking ---------------- #
    def _compute_contributions(self, x_static, x_dynamic):
        """
        Compute both activation-based and gradient-based contributions
        for static and dynamic features.
        """

        # --- Activation-based contribution (feature magnitude) ---
        act_static = self.lin_stat_self(x_static).abs().mean().item()
        act_dynamic = self.lin_dyn_self(x_dynamic).abs().mean().item()

        # --- Gradient-based contribution (learning signal) ---
        grad_static = (
            self.lin_stat_self.weight.grad.abs().mean().item()
            if self.lin_stat_self.weight.grad is not None
            else 0.0
        )
        grad_dynamic = (
            self.lin_dyn_self.weight.grad.abs().mean().item()
            if self.lin_dyn_self.weight.grad is not None
            else 0.0
        )

        # Normalize for comparison (optional)
        total_grad = grad_static + grad_dynamic + 1e-8
        grad_static_ratio = grad_static / total_grad
        grad_dynamic_ratio = grad_dynamic / total_grad

        contrib = {
            "static_dataset": act_static,
            "dynamic_dataset": act_dynamic,
            "grad_static": grad_static,
            "grad_dynamic": grad_dynamic,
            "grad_ratio_static": grad_static_ratio,
            "grad_ratio_dynamic": grad_dynamic_ratio,
        }
        return contrib
    
class StaticDynamicGNN_adv_3(MessagePassing):
    """
    StaticDynamicGNN_adv_3
    ---------------------
    A GNN layer that jointly processes static and dynamic node features
    using explicit MESSAGE → AGGREGATE → UPDATE steps.
    """

    def __init__(self, in_static, in_dynamic, hidden_dim, out_channels, dropout=0.2):
        super().__init__(aggr='add')

        # ---- Self feature transformations ----
        self.lin_stat_self = nn.Linear(in_static, hidden_dim)
        self.lin_dyn_self = nn.Linear(in_dynamic, hidden_dim)

        # ---- Neighbor feature transformations ----
        self.lin_stat_neigh = nn.Linear(in_static, hidden_dim)
        self.lin_dyn_neigh = nn.Linear(in_dynamic, hidden_dim)

        # ----- learnable update -------------------
        self.update_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(hidden_dim))
        
        # ----- Seocnd GNN layer ------------------------
        self.sage = GraphConv(hidden_dim, hidden_dim)

        # ---- Output projection ----
        self.lin_out = nn.Linear(hidden_dim, out_channels)


        # ---- Regularization ----
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

        # ---- Activation ----
        self.act = nn.LeakyReLU(negative_slope=0.1)

        self.reset_parameters()

    # ---------------- Initialization ---------------- #
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ---------------- Message ---------------- #
    def message(self, x_static_j, x_dynamic_j):
        """
        Compute messages from neighboring nodes j → i.
        """
        msg = (
            self.lin_stat_neigh(x_static_j) +
            self.lin_dyn_neigh(x_dynamic_j)
        )
        return msg

    # ---------------- Update ---------------- #
    def update(self, aggr_out, x_static, x_dynamic):
        """
        Combine self-node features with aggregated neighbor messages.
        """

        # Self contribution
        self_emb = (
            self.lin_stat_self(x_static) +
            self.lin_dyn_self(x_dynamic)
        )

        ## Fuse self + neighbors
        # hidden = self_emb + aggr_out

        ## Concatenate instead of summation
        fused = torch.cat([self_emb, aggr_out], dim=-1)
        hidden = self.update_mlp(fused)

        # Normalize + activate + regularize
        # hidden = self.act(hidden)
        # hidden = self.batch_norm(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        return hidden

    # ---------------- Forward ---------------- #
    def forward(self, x_static, x_dynamic, edge_index, return_embedding=False):

        # Message passing (calls message → aggregate → update)
        hidden = self.propagate(
            edge_index,
            x_static=x_static,
            x_dynamic=x_dynamic
        )

        ## ---------- Second GNN layer----------------------
        hidden = self.sage(hidden, edge_index)

        ## ---------- Output layer---------------------------
        out = self.lin_out(hidden)

        contrib_dict = self._compute_contributions(x_static, x_dynamic)

        if return_embedding:
            return hidden, contrib_dict
        return out, contrib_dict

    # ---------------- Contribution Tracking ---------------- #
    def _compute_contributions(self, x_static, x_dynamic):

        act_static = self.lin_stat_self(x_static).abs().mean().item()
        act_dynamic = self.lin_dyn_self(x_dynamic).abs().mean().item()

        grad_static = (
            self.lin_stat_self.weight.grad.abs().mean().item()
            if self.lin_stat_self.weight.grad is not None else 0.0
        )
        grad_dynamic = (
            self.lin_dyn_self.weight.grad.abs().mean().item()
            if self.lin_dyn_self.weight.grad is not None else 0.0
        )

        total_grad = grad_static + grad_dynamic + 1e-8

        return {
            "static_dataset": act_static,
            "dynamic_dataset": act_dynamic,
            "grad_static": grad_static,
            "grad_dynamic": grad_dynamic,
            "grad_ratio_static": grad_static / total_grad,
            "grad_ratio_dynamic": grad_dynamic / total_grad,
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor, Tensor
from typing import Optional

class FusionConv(MessagePassing):
    """
    MessagePassing layer that combines static and dynamic features with separate
    transforms for self and neighbor contributions.

    FusionConv
    -----------------------
    A Graph Neural Network layer that jointly processes static and dynamic node features.
    It integrates feature contributions from self and neighboring nodes using separate
    learnable transformations for static and dynamic modalities, followed by a shared
    hidden representation and output projection.

    Attributes
    ----------
    in_static : int
        Dimensionality of static node features.
    in_dynamic : int
        Dimensionality of dynamic node features.
    hidden_dim : int
        Dimensionality of the intermediate hidden representation.
    out_channels : int
        Number of output channels (task-dependent).

    """
    def __init__(self, in_static: int, in_dynamic: int, out_dim: int, aggr: str = 'add'):
        super().__init__(aggr=aggr)
        # transforms for self (node itself)
        self.lin_stat_self = nn.Linear(in_static, out_dim, bias=False)
        self.lin_dyn_self  = nn.Linear(in_dynamic, out_dim, bias=False)

        # transforms for neighbor (message from neighbors)
        self.lin_stat_neigh = nn.Linear(in_static, out_dim, bias=False)
        self.lin_dyn_neigh  = nn.Linear(in_dynamic, out_dim, bias=False)

        # optional bias and gating
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.lin_stat_self, self.lin_dyn_self, self.lin_stat_neigh, self.lin_dyn_neigh]:
            nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.bias)

    def forward(self,
                x_static: Tensor,
                x_dynamic: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """
        Forward propagation of node features through the message-passing layer.

        Parameters
        ----------
        x_static : torch.Tensor
            Static node feature matrix of shape [N, in_static].
        x_dynamic : torch.Tensor
            Dynamic node feature matrix of shape [N, in_dynamic].
        edge_index : torch.LongTensor
            Graph connectivity in COO format with shape [2, E].

        Returns
        -------
        torch.Tensor
            output tensor.
        """
        # self contributions
        self_msg = self.lin_stat_self(x_static) + self.lin_dyn_self(x_dynamic)

        # neighbor messages computed via propagate -> message -> aggregate
        neigh_msg = self.propagate(edge_index, x_static=x_static, x_dynamic=x_dynamic,
                                   edge_weight=edge_weight)

        out = self_msg + neigh_msg + self.bias
        return out

    def message(self, x_static_j: Tensor, x_dynamic_j: Tensor, edge_weight: OptTensor = None) -> Tensor:
        """
        Message computation for each neighbor j.

        Parameters
        ----------
        x_static_j : torch.Tensor
            Static features of neighboring nodes.
        x_dynamic_j : torch.Tensor
            Dynamic features of neighboring nodes.

        Returns
        -------
        torch.Tensor
            Combined message embedding for each edge.
        """

        # x_j are neighbor features for each edge
        m = self.lin_stat_neigh(x_static_j) + self.lin_dyn_neigh(x_dynamic_j)
        if edge_weight is not None:
            # if graph is weighted, multiply
            return m * edge_weight.view(-1, 1)
        return m

    def __repr__(self):
        return f"{self.__class__.__name__}()"
