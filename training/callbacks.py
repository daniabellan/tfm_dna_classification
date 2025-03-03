import torch

import os

class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_acc):
        if val_acc < self.best_loss - self.min_delta:
            self.best_loss = val_acc
            self.counter = 0
        else:
            self.counter += 1
            print(f"[INFO] Early Stopping {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop  # Retornar el estado de parada temprana

class BestModelCheckpoint:
    def __init__(self, save_dir:str, run_name:str, monitor="loss", mode="min", ):
        self.save_dir = save_dir
        self.run_name = run_name
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else -float("inf")
        self.best_model_path = None

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __call__(self, model, epoch, metrics):
        # Comprueba las metricas del VAL, aunque no se especifique
        current_value = metrics[self.monitor]
        is_better = (current_value < self.best_value) if self.mode == "min" else (current_value > self.best_value)

        if is_better:
            self.best_value = current_value
            model_name = f"{self.run_name}__best_model_epoch_{epoch}.pt"
            self.best_model_path = os.path.join(self.save_dir, model_name)
            torch.save(model.state_dict(), self.best_model_path)
            print(f"Best model saved at epoch {epoch} with {self.monitor}: {current_value:.4f}")

    def get_best_model_path(self):
        return self.best_model_path