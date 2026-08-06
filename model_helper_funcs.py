import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
# ==========================================================
#  Training and Evaluation Utilities
# ==========================================================
def train_epoch(model, optimizer, data, data_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data_mask.train_mask], data.y[data_mask.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, data_mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data_mask.train_mask, data_mask.val_mask, data_mask.test_mask]:
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        accs.append(acc)
    return accs  # train_acc, val_acc, test_acc


def train_loop(model, optimizer, data, data_mask, epochs=200, method_model = "individual_model"):
    train_losses, val_accs, test_accs = [], [], []

    log_dir = os.path.join("runs", f"{method_model}")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, optimizer, data, data_mask)
        train_acc, val_acc, test_acc = evaluate(model, data, data_mask)
        train_losses.append(loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | Test: {test_acc:.3f}")
        # --- Log all key metrics to TensorBoard ---
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

    writer.close()
    return train_losses, val_accs, test_accs



def train_epoch_fusion(model, optimizer, data_static, data_dynamic, data_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data_static.x, data_dynamic.x, data_dynamic.edge_index)
    loss = F.cross_entropy(out[data_mask.train_mask], data_dynamic.y[data_mask.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_fusion(model, data_static, data_dynamic, data_mask):
    model.eval()
    out = model(data_static.x, data_dynamic.x, data_dynamic.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data_mask.train_mask, data_mask.val_mask, data_mask.test_mask]:
        correct = (pred[mask] == data_dynamic.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        accs.append(acc)
    return accs  # train_acc, val_acc, test_acc

def train_loop_fusion(model, optimizer, data_static, data_dynamic, data_mask, epochs=200, method_model = "graph_fusion_experiment"):
    train_losses, val_accs, test_accs, alpha_values = [], [], [], []

    log_dir = os.path.join("runs", f"{method_model}")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, epochs + 1):
        loss = train_epoch_fusion(model, optimizer, data_static, data_dynamic, data_mask)
        train_acc, val_acc, test_acc = evaluate_fusion(model, data_static, data_dynamic, data_mask)
        train_losses.append(loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        alpha = model.alpha.item()
        alpha_values.append(alpha)

        # --- Log all key metrics to TensorBoard ---
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        writer.add_scalar("Fusion/alpha", alpha, epoch)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | Test: {test_acc:.3f} | Alpha: {alpha:.3f}")
    writer.close()
    return train_losses, val_accs, test_accs, alpha_values

###_______________ Used for new models_______________________________________________
def train_epoch_fusion_scra(model, optimizer, data_static, data_dynamic, data_mask):
    model.train()
    optimizer.zero_grad()
    out, alpha = model(data_static.x, data_dynamic.x, data_dynamic.edge_index)
    # # --- FIX START ---
    # train_mask = data_mask.train_mask
    # if train_mask.dtype != torch.bool:
    #     train_mask = train_mask.bool()
    # train_mask = train_mask #.to(out.device)
    # # --- FIX END ---
    loss = F.cross_entropy(out[data_mask.train_mask], data_dynamic.y[data_mask.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_fusion_scra(model, data_static, data_dynamic, data_mask):
    model.eval()
    out,alpha = model(data_static.x, data_dynamic.x, data_dynamic.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data_mask.train_mask, data_mask.val_mask, data_mask.test_mask]:
        correct = (pred[mask] == data_dynamic.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        accs.append(acc)
    accs.append(alpha)
    return accs # train_acc, val_acc, test_acc

def train_loop_fusion_scra(model, optimizer, data_static, data_dynamic, data_mask, epochs=200, method_model = "graph_fusion_experiment"):
    train_losses, val_accs, test_accs, alpha_values = [], [], [], []

    log_dir = os.path.join("runs", f"{method_model}")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, epochs + 1):
        loss = train_epoch_fusion_scra(model, optimizer, data_static, data_dynamic, data_mask)
        train_acc, val_acc, test_acc , alpha= evaluate_fusion_scra(model, data_static, data_dynamic, data_mask)
        train_losses.append(loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        # alpha = model.alpha.item()
        # alpha_values.append(alpha)

        # After computing alpha somewhere in the model:
        if isinstance(alpha, torch.Tensor):
            alpha_mean = alpha.mean().item()
            alpha_std = alpha.std().item()
        elif isinstance(alpha, dict):
            alpha_mean = {}
            alpha_std = {}
            for key, value in alpha.items():
                if isinstance(value, torch.Tensor):
                    alpha_mean[key] = value.mean().item()
                    alpha_std[key] = value.std().item()
                else:
                    alpha_mean[key] = float(value)
                    alpha_std[key] = 0.0
        else:
            alpha_mean = float(alpha)
            alpha_std = 0.0
        
        alpha_values.append(alpha_mean)

        # --- Log all key metrics to TensorBoard ---
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        # writer.add_scalar("Fusion/alpha", alpha_mean, epoch)

        if isinstance(alpha, dict):
            if epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                    f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | Test: {test_acc:.3f} | Alpha_stat: {alpha_mean['static_dataset']:.3f} | Alpha_dyn: {alpha_mean['dynamic_dataset']:.3f}")
        else:
            if epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                    f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | Test: {test_acc:.3f} | Alpha: {alpha_mean:.3f}")
    writer.close()
    return train_losses, val_accs, test_accs, alpha_values

#-------------------- START: training and evaluation functions for M6 and M7 (Sliding Window & Late fusion appraoch)

# from torch.utils.data import 
# from torch_geometric.loader import DataLoader
def train_epoch_in(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    # criterion = nn.MSELoss()

    for batch in loader:
        optimizer.zero_grad()

        preds, _ = model(
            batch.x_static.to(device),
            batch.x_dynamic.to(device),
            batch.edge_index.to(device)
        )
        # print(batch.x_dynamic.shape)
        loss = criterion(preds, batch.y.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_in(model, loader, criterion, device):
    model.eval()
    # criterion = nn.MSELoss()
    total_loss = 0.0

    for batch in loader:
        preds, _ = model(
            batch.x_static.to(device),
            batch.x_dynamic.to(device),
            batch.edge_index.to(device)
        )
        # breakpoint()

        loss = criterion(preds, batch.y.to(device))
        total_loss += loss.item()

    return total_loss / len(loader)

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@torch.no_grad()
def evaluate_regression(model, loader, device):
    model.eval()

    all_preds = []
    all_targets = []
    huber_loss = torch.nn.SmoothL1Loss()

    for batch in loader:
        preds, _ = model(
            batch.x_static.to(device),
            batch.x_dynamic.to(device),
            batch.edge_index.to(device)
        )

        all_preds.append(preds.detach().cpu())
        all_targets.append(batch.y.detach().cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    huber = huber_loss(all_targets, all_preds)
    

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "huber":huber,
    }

@torch.no_grad()
def evaluate_and_collect(model, loader, device, bdynamiconly = False):
    model.eval()

    all_preds = []
    all_targets = []
    huber_loss = torch.nn.SmoothL1Loss()

    # For spatial analysis
    preds_per_timestep = []
    targets_per_timestep = []

    for batch in loader:

        if not bdynamiconly:
            preds, _ = model(
                batch.x_static.to(device),
                batch.x_dynamic.to(device),
                batch.edge_index.to(device)
                
            )
        else:
            preds, _ = model(
                batch.x_dynamic.to(device),
                batch.edge_index.to(device)
            )

        preds = preds.squeeze()
        # preds = torch.sigmoid(preds)
        

        preds_cpu = preds.detach().cpu()
        targets_cpu = batch.y.detach().cpu()

        all_preds.append(preds_cpu)
        all_targets.append(targets_cpu)

        # Save per timestep
        preds_per_timestep.append(preds_cpu.numpy())
        targets_per_timestep.append(targets_cpu.numpy())

    # ---- GLOBAL METRICS ----
    # all_preds_np = torch.cat(all_preds).numpy()
    # all_targets_np = torch.cat(all_targets).numpy()

    # mse = mean_squared_error(all_targets_np, all_preds_np)
    # rmse = np.sqrt(mse)
    # mae = mean_absolute_error(all_targets_np, all_preds_np)
    # r2 = r2_score(all_targets_np, all_preds_np)
    # huber = huber_loss(all_targets, all_preds)
    
    all_preds_tensor = torch.cat(all_preds)
    all_targets_tensor = torch.cat(all_targets)

    all_preds_np = all_preds_tensor.numpy()
    all_targets_np = all_targets_tensor.numpy()

    mse = mean_squared_error(all_targets_np, all_preds_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets_np, all_preds_np)
    r2 = r2_score(all_targets_np, all_preds_np)

    huber = huber_loss(all_preds_tensor, all_targets_tensor).item()


    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "huber":huber,

    }

    return metrics, preds_per_timestep, targets_per_timestep



#--------------------  END: training and evaluation functions for M6 and M7 (Sliding Window & Late fusion appraoch)

##-------------START: Predicted Time series analysis----------

from collections import defaultdict

def build_adj(edge_index):
    adj = defaultdict(set)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    for s, d in zip(src, dst):
        adj[s].add(d)
        adj[d].add(s)  # remove if graph is directed

    return adj


def get_neighbors(node, adj, order=1):
    visited = set([node])
    frontier = set([node])

    for _ in range(order):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(adj[n])
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier

    visited.remove(node)
    return list(visited)

##-------------END: Predicted Time series analysis----------