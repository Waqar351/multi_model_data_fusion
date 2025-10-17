# from torch_geometric.nn import GCNConv
# import torch
# import torch.nn.functional as F


# ## Define 2-Layer GCN Model
# # ==========================================================
# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)
#         self.dropout = dropout

#     def forward(self, x, edge_index, return_embedding=False):
#         # Layer 1
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         # Save embedding before classification
#         x_emb = x.clone()

#         # Layer 2: classification logits
#         x_out = self.conv2(x, edge_index)

#         if return_embedding:
#             return x_emb  # return latent space before final logits
#         return x_out
    
#     # ==========================================================
#     #  Training, Validation, and Testing
#     # ==========================================================
#     def train(model, optimizer, data):
#         model.train()
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index)
#         loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
#         loss.backward()
#         optimizer.step()
#         return loss.item()
    
#     @torch.no_grad()
#     def evaluate(model, data):
#         model.eval()
#         out = model(data.x, data.edge_index)
#         pred = out.argmax(dim=1)
#         accs = []
#         for mask in [data.train_mask, data.val_mask, data.test_mask]:
#             correct = (pred[mask] == data.y[mask]).sum()
#             acc = int(correct) / int(mask.sum())
#             accs.append(acc)
#         return accs  # train_acc, val_acc, test_acc
    
#     def train_loop(model, optimizer, data, epochs = 200):
#         # ==========================================================
#         #  Run Training Loop
#         # ==========================================================
#         train_losses, val_accs = [], []

#         for epoch in range(1, epochs):
#             loss = GCN.train(model, optimizer, data)
#             train_acc, val_acc, test_acc = GCN.evaluate(model, data)
#             train_losses.append(loss)
#             val_accs.append(val_acc)

#             if epoch % 20 == 0:
#                 print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
#                     f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | Test: {test_acc:.3f}")
        
#         return train_losses, val_accs

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ==========================================================
# Define 2-Layer GCN Model
# ==========================================================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
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


# ==========================================================
#  Training and Evaluation Utilities
# ==========================================================
def train_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        accs.append(acc)
    return accs  # train_acc, val_acc, test_acc


def train_loop(model, optimizer, data, epochs=200):
    train_losses, val_accs = [], []

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, optimizer, data)
        train_acc, val_acc, test_acc = evaluate(model, data)
        train_losses.append(loss)
        val_accs.append(val_acc)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | Test: {test_acc:.3f}")

    return train_losses, val_accs
