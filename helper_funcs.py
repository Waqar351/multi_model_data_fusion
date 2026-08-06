import pandas as pd
import re
import torch
# Defining self loops in graph
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data


def label_crime_nodes(df, threshold=5):
    """
    Generate a binary crime label for each node based on monthly crime counts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing monthly crime counts per node.
    threshold : int or float
        Threshold for labeling a node as 'crime' (1) or 'no crime' (0).
    
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added 'total_crimes' and 'crime_label' columns.
    """
    # Detect columns that match the pattern YYYY.MM (e.g., 2006.01, 2017.12)
    month_cols = [col for col in df.columns if re.match(r'^\d{4}\.\d{2}$', col)]
    
    # breakpoint()
    if not month_cols:
        raise ValueError("No month columns found in format YYYY.MM")
    
    # Compute total crime count across all months
    df['total_crimes'] = df[month_cols].sum(axis=1)
    
    # Create binary label: 1 = crime, 0 = no crime
    df['crime_label'] = (df['total_crimes'] > threshold).astype(int)
    
    return df

def mapping_edge_index(edge_index, mapping_dict):
    # Convertir a numpy para operaciones más eficientes
    edges_np = edge_index.numpy()

    # Vectorizar el mapeo usando pandas (más rápido que un bucle)
    df_edges = pd.DataFrame(edges_np)
    edges_mapeados = df_edges.applymap(lambda x: mapping_dict.get(x, x)).values

    # Convertir de vuelta a tensor
    return torch.tensor(edges_mapeados, dtype=torch.long)

def map_edges_new_index(df_nodes, df_edges ):
    
    # Step 1: Get unique node IDs (from your main crime dataframe)
    unique_ids = df_nodes['Nodo'].unique()

    # Step 2: Create a new DataFrame for mapping
    mapping = pd.DataFrame({
        'ID_Original': unique_ids,
        'ID_Equivalente': range(len(unique_ids))  # Assign consecutive integers
    })

    dict_mapeo = dict(zip(mapping['ID_Original'], mapping['ID_Equivalente']))


    ###############################

    # Create a boolean mask
    mask = df_edges['Nodo1'].isin(df_nodes['Nodo']) & df_edges['Nodo2'].isin(df_nodes['Nodo'])

    # Filter df_edges using the mask
    df_edges_filtrado = df_edges[mask]

    df_edges = df_edges_filtrado


    # Extract columns 'Nodo1' and 'Nodo2'
    edges = df_edges[['Nodo1', 'Nodo2']].values

    # Convert to a PyTorch tensor
    edges_tensor = torch.tensor(edges, dtype=torch.long)

    # Transpose the tensor to have shape (2, num_edges)
    edges_tensor = edges_tensor.t().contiguous()

    print(edges_tensor)
    edges_tensor.shape

    # Aplicar el mapeo
    edge_index_mapeado = mapping_edge_index(edges_tensor, dict_mapeo)
    print("edge_index_mapeado -------> ", edge_index_mapeado)

    edge_index = edge_index_mapeado

    num_nodes = num_nodes = len(unique_ids)


    

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_index = to_undirected(edge_index)

    return edge_index

# ==========================================================
# 1. Reproducibility Setup
# ==========================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensures deterministic behavior in CUDA (may slow things)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------
# 7. Utility function: prepare data dynamically
# ----------------------------------------------------------
def prepare_data(mode, data_static, data_dynamic):
    """Return Data object based on chosen mode."""
    if mode == "static":
        print("➡️ Using static features only.")
        return data_static
    elif mode == "dynamic":
        print("➡️ Using dynamic features only.")
        return data_dynamic
    # elif mode == "separate":
    #     print("➡️ Using dynamic features only.")
    #     return data_static, data_dynamic
    elif mode == "both":
        print("➡️ Using combined static + dynamic features.")
        combined_x = torch.cat([data_static.x, data_dynamic.x], dim=1)
        data_combined = Data(
            x=combined_x,
            edge_index=data_static.edge_index,
            y=data_static.y
        )
        # copy masks later
        return data_combined
    else:
        raise ValueError(f"Unknown data mode: {mode}")
    
from sklearn.model_selection import train_test_split
import numpy as np
import torch

def stratified_graph_split(labels, train_ratio=0.6, val_ratio=0.2, random_state=42):
    """
    Create stratified boolean masks (train/val/test) for graph data.
    Works for both binary and multi-class labels.
    
    Args:
        labels (array-like): Node labels (numpy or torch tensor)
        train_ratio (float): Portion of training samples (default=0.6)
        val_ratio (float): Portion of validation samples (default=0.2)
        random_state (int): Random seed
        
    Returns:
        train_mask, val_mask, test_mask (torch.BoolTensor)
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    num_nodes = len(labels)
    indices = np.arange(num_nodes)
    
    # --- Train vs Temp split (Temp = Val + Test)
    train_indices, temp_indices, y_train, y_temp = train_test_split(
        indices,
        labels,
        stratify=labels,
        test_size=(1 - train_ratio),
        random_state=random_state
    )
    
    # --- Validation vs Test split (stratified within Temp)
    val_size = val_ratio / (1 - train_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        stratify=y_temp,
        test_size=(1 - val_size),
        random_state=random_state
    )
    
    # --- Boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    return train_mask, val_mask, test_mask


from torch_geometric.data import Data, Dataset

# class CrimeWindowPyGDataset(Dataset):
#     def __init__(self, x_static, x_dynamic, edge_index,
#                  window_size, start_t, end_t):
#         super().__init__()
#         self.x_static = x_static
#         self.x_dynamic = x_dynamic
#         self.edge_index = edge_index
#         self.window_size = window_size
#         self.start_t = start_t
#         self.end_t = end_t

#     def len(self):
#         return (self.end_t - self.start_t) - self.window_size

#     def get(self, idx):
#         t = self.start_t + idx

#         x_dyn_window = self.x_dynamic[:, t:t+self.window_size]
#         y = self.x_dynamic[:, t+self.window_size]

#         # Concatenate static + dynamic window
#         # x = torch.cat([self.x_static, x_dyn_window], dim=1)

#         return Data(
#             x_static=self.x_static,
#             x_dynamic =x_dyn_window,
#             edge_index=self.edge_index,
#             y=y,
#             num_nodes = self.x_static.shape[0]
#         )
import torch
from torch_geometric.data import Dataset, Data

class CrimeWindowPyGDataset(Dataset):
    def __init__(self, x_static, x_dynamic, edge_index,
                 window_size, start_t, end_t):
        super().__init__()

        self.x_static = x_static            # (N, F_static)
        self.x_dynamic = x_dynamic          # (N, T)
        self.edge_index = edge_index
        self.window_size = window_size
        self.start_t = start_t
        self.end_t = end_t

        self.num_nodes = x_static.shape[0]

    def len(self):
        return (self.end_t - self.start_t) - self.window_size

    def get(self, idx):

        t = self.start_t + idx

        # Dynamic window: shape (N, window_size)
        x_dyn_window = self.x_dynamic[:, t:t+self.window_size]

        # Target: next month crime
        y = self.x_dynamic[:, t+self.window_size]

        # Node IDs (0 ... N-1)
        node_ids = torch.arange(self.num_nodes)

        # Time ID for this window (same for all nodes)
        time_id = torch.full((self.num_nodes,), t)

        return Data(
            x_static=self.x_static,
            x_dynamic=x_dyn_window,
            edge_index=self.edge_index,
            y=y,
            node_ids=node_ids,
            time_id=time_id,
            num_nodes=self.num_nodes
        )

    
#________------ Contribution analysis functions starts -------____________________________

def modality_contribution(z_static, z_dynamic, eps=1e-8):
    """
    z_static: [N, h]
    z_dynamic: [N, h]
    """
    static_norm = torch.norm(z_static, p=2, dim=1).mean()
    dynamic_norm = torch.norm(z_dynamic, p=2, dim=1).mean()

    total = static_norm + dynamic_norm + eps

    return {
        "static": (static_norm / total).item(),
        "dynamic": (dynamic_norm / total).item()
    }

def conditional_modality_contribution(z_static, z_dynamic, x_dynamic, eps=1e-8):
    """
    Only consider nodes where dynamic signal exists.
    """
    mask = (x_dynamic.abs().sum(dim=1) > 0.95)

    if mask.sum() == 0:
        return {"static": 0.0, "dynamic": 0.0}

    static_norm = torch.norm(z_static[mask], p=2, dim=1).mean()
    dynamic_norm = torch.norm(z_dynamic[mask], p=2, dim=1).mean()

    total = static_norm + dynamic_norm + eps

    return {
        "static": (static_norm / total).item(),
        "dynamic": (dynamic_norm / total).item()
    }

def gradient_modality_contribution(
    model, batch, criterion, device="cpu", eps=1e-8
):
    model.train()  # gradients must be enabled
    model.zero_grad()

    batch = batch.to(device)

    # ---- Forward encoders
    z_static = model.late_fusion.static_encoder(
        batch.x_static, batch.edge_index
    )
    z_dynamic = model.late_fusion.dynamic_encoder(
        batch.x_dynamic, batch.edge_index
    )

    # Retain gradients on non-leaf tensors
    z_static.retain_grad()
    z_dynamic.retain_grad()

    # ---- Forward fusion
    z_fused = torch.cat([z_static, z_dynamic], dim=-1)
    # z_fused = model.late_fusion.sage(z_fused, batch.edge_index)  # [N, h]
    out = model.late_fusion.fusion_fc(z_fused).squeeze(-1)

    loss = criterion(out, batch.y)
    loss.backward()

    # ---- Now gradients exist
    grad_static = z_static.grad.norm(p=2)
    grad_dynamic = z_dynamic.grad.norm(p=2)

    total = grad_static + grad_dynamic + eps

    return {
        "static": (grad_static / total).item(),
        "dynamic": (grad_dynamic / total).item()
    }

@torch.no_grad()
def ablation_evaluate(model, loader, criterion, device="cpu", mode="static"):
    model.eval()
    losses = []

    for batch in loader:
        batch = batch.to(device)

        if mode == "static":
            x_dynamic = torch.zeros_like(batch.x_dynamic)
            x_static = batch.x_static
        elif mode == "dynamic":
            x_static = torch.zeros_like(batch.x_static)
            x_dynamic = batch.x_dynamic
        else:
            raise ValueError("mode must be 'static' or 'dynamic'")

        out, _ = model(x_static, x_dynamic, batch.edge_index)
        # loss = F.mse_loss(out, batch.y)
        loss = criterion(out, batch.y)
        losses.append(loss.item())

    return sum(losses) / len(losses)

def unpack_contribution(hist):
    static = [h["static"] for h in hist]
    dynamic = [h["dynamic"] for h in hist]
    return static, dynamic




#________------ Contribution analysis functions ends -------____________________________

