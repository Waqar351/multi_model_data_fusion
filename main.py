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


from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv

import sys
from helper_funcs import *
from plots import *
from model import *


set_seed(42)
####################
loss_func = "MSE"  # "MSE" (Mean Square Error)   # "MAE" (Mean Absolute Error) #Huber
data_set = "real" # "real"  "artificial"    
type_graph_conv = "SAGE"# "SAGE" 
####################

if(data_set =="artificial"):
    print("artificial data")
    name_file_dataset = os.path.join("datasets", "artificial_data_12_clusters_full.csv")
else: # real dataset
    print("real data")
    name_file_dataset = "datasets/datasubset_nodes_waqar.csv"

df_nodes = pd.read_csv(name_file_dataset) # orignal raw data

df_edges = pd.read_csv('datasets/aristas_subgrafoSPdaily.csv')

edge_index = map_edges_new_index(df_nodes, df_edges)

####_______________________________________________________________________________________________________

df_nodes_label = label_crime_nodes(df_nodes, threshold= 5)
crime_label_col = 'crime_label'
# plot_crime_vs_nocrime_ratio(df_nodes_label)

dynamic_cols = [col for col in df_nodes_label.columns if re.match(r'^\d{4}\.\d{2}$', col)]

# selected_cols = list(dynamic_cols) + [crime_label_col]

dynamic_dt = df_nodes_label[dynamic_cols]

cols_to_drop = dynamic_cols + ['Nodo', 'Pontos_de_onibus', 'lat', 'long', 'total_crimes', 'crime_label']
static_dt = df_nodes_label.drop(columns= cols_to_drop)
# breakpoint()
dynamic_dt_smooth = dynamic_dt.apply(lambda row: row.ewm(alpha=0.1).mean(), axis=1)
dynamic_dt_smooth_log = np.log1p(np.array(dynamic_dt_smooth))

# Compute global min and max
global_min = dynamic_dt_smooth_log.min().min()
global_max = dynamic_dt_smooth_log.max().max()

# Apply global min-max normalization
dynamic_dt_norm = pd.DataFrame((dynamic_dt_smooth_log - global_min) / (global_max - global_min))
dynamic_dt_norm

scaler = MinMaxScaler()
static_dt_norm = pd.DataFrame(scaler.fit_transform(static_dt))


# breakpoint()

#___ Convert into Tensors and then graphs_________
# Create graph
# x_tensor = torch.tensor(dynamic_dt_norm.values, dtype=torch.float32)
x_tensor = torch.tensor(static_dt_norm.values, dtype=torch.float32)
# breakpoint()
x_tensor = torch.nan_to_num(x_tensor, nan=0.0)

crime_labels = df_nodes_label['crime_label']
crime_label_tensor = torch.tensor(crime_labels.values, dtype=torch.long)

data = Data(x=x_tensor, edge_index=edge_index, y = crime_label_tensor)
# breakpoint()
# ==========================================================
# 3. Train / Validation / Test split
# ==========================================================
num_nodes = data.num_nodes
num_train = int(0.6 * num_nodes)
num_val = int(0.2 * num_nodes)
num_test = num_nodes - num_train - num_val

num_features = data.x.shape[1]
num_classes = 2

perm = torch.randperm(num_nodes)

data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

data.train_mask[perm[:num_train]] = True
data.val_mask[perm[num_train:num_train+num_val]] = True
data.test_mask[perm[num_train+num_val:]] = True

# ==========================================================
#  Initialize model and optimizer
# ==========================================================
model = GCN(in_channels=num_features, hidden_channels=32, out_channels=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# ==========================================================
#  Training, Validation, and Testing
# ==========================================================
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        accs.append(acc)
    return accs  # train_acc, val_acc, test_acc

# ==========================================================
#  Run Training Loop
# ==========================================================
train_losses, val_accs = [], []

for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = evaluate()
    train_losses.append(loss)
    val_accs.append(val_acc)

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
              f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | Test: {test_acc:.3f}")

# ==========================================================
# 8. Plot Training Curves
# ==========================================================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss', color='royalblue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accs, label='Validation Accuracy', color='darkorange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Curve')
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# Final Evaluation
# ==========================================================
train_acc, val_acc, test_acc = evaluate()
print(f"Final Results:\nTrain Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | Test Acc: {test_acc:.3f}")

# ==========================================================
# Extract Node Embeddings (Latent Space)
# ==========================================================
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index, return_embedding=True)

print(f"Node Embeddings shape: {embeddings.shape}")  # [num_nodes, out_channels]

# breakpoint()
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings.cpu())

df_emb = pd.DataFrame(emb_2d, columns=['x', 'y'])
df_emb['label'] = data.y.cpu().numpy()

plt.figure(figsize=(6,5))
sns.scatterplot(data=df_emb, x='x', y='y', hue='label', palette='Set1', alpha=0.8)
plt.title("t-SNE Visualization of GNN Node Embeddings")
plt.show()
breakpoint()

