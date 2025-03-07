import torch
import numpy as np

class EarlyStopping:
    """
    Implements early stopping to halt training when validation loss stops improving.

    Args:
        patience (int): Number of epochs to wait before stopping if no improvement.
        min_delta (float): Minimum change in validation loss to be considered as improvement.
        verbose (bool): If True, prints early stopping messages.
        save_path (str, optional): Path to save the best model. If None, no model is saved.
    """
    def __init__(self, patience:int=5, min_delta:float=0.001, verbose:bool=True, save_path:str=None):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.save_path = save_path
        self.best_loss = np.inf
        self.counter = 0

    def __call__(self, val_loss, model):
        """
        Checks if validation loss improved. If not, increments counter and stops training if patience is reached.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Model to save if improvement is detected.
        
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}. Resetting patience.")
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)  # Save the best model
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss. Patience counter: {self.counter}/{self.patience}")

        return self.counter >= self.patience  # Stop training if patience is exceeded
