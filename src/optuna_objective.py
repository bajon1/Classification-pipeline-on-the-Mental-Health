import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold

from .model import MLP
from .dataset import build_fold_dataloaders
from .train import train_fold

def _make_model(trial, input_size, output_size):
    n_hidden = trial.suggest_int("n_hidden", 1, 16)
    optuna_hidden_layers = [
        trial.suggest_categorical(f"hidden_dim_l{i}", [16, 32, 64, 128, 256])
        for i in range(n_hidden)
    ]
    optuna_dropout_p = trial.suggest_float("dropout_p", 0.1, 0.9)
    optuna_activation = trial.suggest_categorical("activation", ["relu", "tanh"])

    return MLP(input_size=input_size, output_size=output_size,
               hidden_layers=optuna_hidden_layers,
               dropout_p=optuna_dropout_p, activation=optuna_activation)

def _make_optimizer(trial, optuna_model):
    optuna_optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    optuna_lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    optuna_optimizer = getattr(torch.optim, optuna_optimizer_name)

    return optuna_optimizer(optuna_model.parameters(), lr=optuna_lr)

def make_objective(X, y, weights, input_size, output_size, n_epochs=10 , n_splits=5, device='cpu'):
    def objective(trial):
        optuna_batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
        criterion = nn.CrossEntropyLoss()

        fold_val_losses = []

        for fold_idx in range(n_splits):
            optuna_model = _make_model(trial, input_size, output_size)
            optuna_optimizer = _make_optimizer(trial, optuna_model)

            train_loader, val_loader = build_fold_dataloaders(X=X, y=y, weights=weights,
                                                              n_splits=n_splits, fold_idx=fold_idx,
                                                              batch_size=optuna_batch_size)

            history = train_fold(model=optuna_model, train_loader=train_loader,
                                 val_loader=val_loader, optimizer=optuna_optimizer,
                                 criterion=criterion, fold_idx=fold_idx,
                                 device=device, n_epochs=n_epochs, write_model=True,
                                 tensorboard_dir=f"../runs/tensorboard/optuna",
                                 checkpoint_dir=f"../runs/checkpoints/optuna",
                                 csv_dir=f"../runs/csv_logs/optuna")

            fold_val_losses.append(history['best_val_loss'])

        return sum(fold_val_losses)/len(fold_val_losses)

    return objective