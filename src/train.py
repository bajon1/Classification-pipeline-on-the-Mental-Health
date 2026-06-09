import os
import csv
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def _train_one_epoch(model, optimizer, criterion, train_loader, clip_grad_norm, device='cpu'):
    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        train_loss += loss.item()*len(X_batch)
    train_loss /= len(train_loader.dataset)

    return train_loss

@torch.no_grad()
def _validate(model, val_loader, criterion, device='cpu'):
    model.eval()
    val_loss = 0

    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        loss = criterion(model(X_batch), y_batch)
        val_loss += loss.item()*len(X_batch)
    val_loss /= len(val_loader.dataset)

    return val_loss

def train_fold(model, train_loader, val_loader, optimizer, criterion, fold_idx=0,
          device='cpu', n_epochs=100, clip_grad_norm=1.0,  write_model=True,
          tensorboard_dir="../runs/tensorboard", checkpoint_dir="../runs/checkpoints",
          csv_dir="../runs/csv_logs"):

    fold_name = f"fold_{fold_idx}"
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, fold_name))

    with open(os.path.join(csv_dir, f"{fold_name}_losses.csv"), "w", newline='') as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    model.to(device)
    best_model_state = None
    best_val_loss = np.inf
    history = {'train_losses': [], 'val_losses': []}

    for epoch in range(n_epochs):
        train_loss = (_train_one_epoch(model, optimizer, criterion, train_loader, clip_grad_norm, device))
        val_loss = (_validate(model, val_loader, criterion, device))

        writer.add_scalars("Epoch losses:", {"train_loss": train_loss, "val_loss": val_loss}, epoch)
        writer.flush()

        with open(os.path.join(csv_dir, f"{fold_name}_losses.csv"), "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss])

        if val_loss <  best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            if write_model:
                torch.save({"epoch": epoch, "model_state_dict": best_model_state, "best_val_loss": best_val_loss},
                 os.path.join(checkpoint_dir, f"{fold_name}_best.pt"))

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['best_model_state'] = best_model_state
        history['best_val_loss'] = best_val_loss

        print(f"[Fold {fold_idx}] Epoch {epoch:>3}/{n_epochs} | train: {train_loss:.4f} | val: {val_loss:.4f}"
              + (" <- new best" if val_loss == best_val_loss else ""))

    writer.close()
    return history