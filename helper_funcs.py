import pandas as pd
import re
import torch
# Defining self loops in graph
from torch_geometric.utils import add_self_loops
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
