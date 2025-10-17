import torch.nn.functional as F
import torch

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