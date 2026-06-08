from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
import torch


def build_fold_dataloaders(X, y, weights, fold_idx, batch_size, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    splits = list(skf.split(X, y))
    train_idx, val_idx = splits[fold_idx]

    X_train, X_val = X[train_idx].float(), X[val_idx].long()
    y_train, y_val = y[train_idx].float(), y[val_idx].long()

    fold_weights = weights[train_idx]
    sampler = WeightedRandomSampler(fold_weights, len(fold_weights), replacement=True)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                                batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=batch_size)

    return train_loader, val_loader