import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def _train_one_epoch(model, optimizer, criterion, train_loader, device='cpu'):
    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()*len(X_batch)
    train_loss /= len(train_loader)

    return train_loss

@torch.no_grad()
def _validate(model, val_loader, criterion, epoch):
    model.eval()
    val_loss = 0

    for X_batch, y_batch in val_loader:
        loss = criterion(model(X_batch), y_batch)
        val_loss += loss.item()*len(X_batch)
    val_loss /= len(val_loader)

    return val_loss

def train(model, train_loader, val_loader, optimizer, criterion, fold_idx=0,
          device='cpu', n_epochs=100, clip_grad_norm=1.0,  write_model=True,
          tensorboard_dir="../runs/tensorboard"):
    history = {'train_losses': [], 'val_losses': []}
    best_val_loss = np.inf

    for epoch in range(n_epochs):
        train_loss = (_train_one_epoch(model, optimizer, criterion, train_loader))
        val_loss = (_validate(model, val_loader, criterion, epoch))

        if val_loss <  best_val_loss:
            best_val_loss = val_loss

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)

    return history