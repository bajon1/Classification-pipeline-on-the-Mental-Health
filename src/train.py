import torch
import torch.optim as optim

def _train_one_epoch(model, optimizer, criterion, train_loader, epoch):
    model.train()
    pass

def _evaluate(model, val_loader, criterion, epoch):
    pass

def train(model, train_loader, val_loader, optimizer, criterion, fold_idx=0, device='cpu', n_epochs=10):
    pass