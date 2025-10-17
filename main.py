import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import optuna
import pandas as pd
import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import SAGEConv, GATConv
import sys
from helper_funcs import *
from plots import *
from models import *

# ----------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------
set_seed(42)

loss_func = "MSE"        # "MSE" | "MAE" | "Huber"
data_set = "real"         # "real" | "artificial"
type_graph_conv = "SAGE"  # "SAGE" | "GCN" | "GAT"
data_mode = "both"        # "static" | "dynamic" | "both"
print(f"Selected data mode: {data_mode}")

# ----------------------------------------------------------
# 2. Load dataset
# ----------------------------------------------------------
if data_set == "artificial":
    print("Using artificial dataset")
    name_file_dataset = os.path.join("datasets", "artificial_data_12_clusters_full.csv")
else:
    print("Using real dataset")
    name_file_dataset = "datasets/datasubset_nodes_waqar.csv"

df_nodes = pd.read_csv(name_file_dataset)
df_edges = pd.read_csv("datasets/aristas_subgrafoSPdaily.csv")
edge_index = map_edges_new_index(df_nodes, df_edges)

# ----------------------------------------------------------
# 3. Label nodes
# ----------------------------------------------------------
df_nodes_label = label_crime_nodes(df_nodes, threshold=5)
crime_label_col = "crime_label"

# ----------------------------------------------------------
# 4. Extract feature groups
# ----------------------------------------------------------
dynamic_cols = [col for col in df_nodes_label.columns if re.match(r'^\d{4}\.\d{2}$', col)]
dynamic_dt = df_nodes_label[dynamic_cols]
cols_to_drop = dynamic_cols + ['Nodo', 'Pontos_de_onibus', 'lat', 'long', 'total_crimes', 'crime_label']
static_dt = df_nodes_label.drop(columns=cols_to_drop)

# ----------------------------------------------------------
# 5. Normalize features
# ----------------------------------------------------------
# Dynamic (time series)
dynamic_dt_smooth = dynamic_dt.apply(lambda row: row.ewm(alpha=0.1).mean(), axis=1)
dynamic_dt_smooth_log = np.log1p(np.array(dynamic_dt_smooth))

global_min = dynamic_dt_smooth_log.min().min()
global_max = dynamic_dt_smooth_log.max().max()
dynamic_dt_norm = pd.DataFrame((dynamic_dt_smooth_log - global_min) / (global_max - global_min))

# Static
scaler = MinMaxScaler()
static_dt_norm = pd.DataFrame(scaler.fit_transform(static_dt))

# ----------------------------------------------------------
# 6. Convert to tensors
# ----------------------------------------------------------
dynamic_tensor = torch.tensor(dynamic_dt_norm.values, dtype=torch.float32)
dynamic_tensor = torch.nan_to_num(dynamic_tensor, nan=0.0)

static_tensor = torch.tensor(static_dt_norm.values, dtype=torch.float32)
static_tensor = torch.nan_to_num(static_tensor, nan=0.0)

crime_labels = df_nodes_label[crime_label_col]
crime_label_tensor = torch.tensor(crime_labels.values, dtype=torch.long)

data_dynamic = Data(x=dynamic_tensor, edge_index=edge_index, y=crime_label_tensor)
data_static = Data(x=static_tensor, edge_index=edge_index, y=crime_label_tensor)

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

# ----------------------------------------------------------
# 8. Train / Validation / Test split
# ----------------------------------------------------------
num_nodes = df_nodes_label.shape[0]
num_train = int(0.6 * num_nodes)
num_val = int(0.2 * num_nodes)
num_test = num_nodes - num_train - num_val

perm = torch.randperm(num_nodes)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[perm[:num_train]] = True
val_mask[perm[num_train:num_train+num_val]] = True
test_mask[perm[num_train+num_val:]] = True

# ----------------------------------------------------------
# 9. Select data mode and attach masks
# ----------------------------------------------------------
data_selected = prepare_data(data_mode, data_static, data_dynamic)
data_selected.train_mask = train_mask
data_selected.val_mask = val_mask
data_selected.test_mask = test_mask

num_features = data_selected.x.shape[1]
num_classes = 2

# ----------------------------------------------------------
# 10. Model initialization
# ----------------------------------------------------------
model = GCN(in_channels=num_features, hidden_channels=32, out_channels=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# ----------------------------------------------------------
# 11. Train model
# ----------------------------------------------------------
train_losses, val_accs = train_loop(model, optimizer, data_selected, epochs=200)
plot_train_val_curves(train_losses, val_accs)

# ----------------------------------------------------------
# 12. Evaluate model
# ----------------------------------------------------------
train_acc, val_acc, test_acc = evaluate(model, data_selected)
print(f"Final Results:\nTrain Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | Test Acc: {test_acc:.3f}")

# ----------------------------------------------------------
# 13. Extract embeddings
# ----------------------------------------------------------
model.eval()
with torch.no_grad():
    embeddings = model(data_selected.x, data_selected.edge_index, return_embedding=True)

print(f"Node Embeddings shape: {embeddings.shape}")

# ----------------------------------------------------------
# 14. t-SNE visualization
# ----------------------------------------------------------
from sklearn.manifold import TSNE
import seaborn as sns

tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings.cpu())

df_emb = pd.DataFrame(emb_2d, columns=['x', 'y'])
df_emb['label'] = data_selected.y.cpu().numpy()

plt.figure(figsize=(6,5))
sns.scatterplot(data=df_emb, x='x', y='y', hue='label', palette='Set1', alpha=0.8)
plt.title(f"t-SNE Visualization of GNN Node Embeddings ({data_mode})")
plt.show()

