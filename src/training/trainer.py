import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from src.training.train import train_one_epoch
from src.training.validate import validate
from src.utils.common import print_epoch_summary
from src.training.early_stopping import EarlyStopping

class Trainer:
    """
    A Trainer class to handle model training, validation, and testing.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for testing data.
        training_config (dict): Training configuration.
        device (torch.device): Target device (CPU/GPU).
    """

    def __init__(self, model, train_loader, val_loader, test_loader, training_config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.training_config = training_config

        # Initialize training components
        self.criterion = self.initialize_criterion()
        self.optimizer, self.scheduler = self.initialize_optimizer_scheduler()
        self.early_stopping = self.initialize_early_stopping()

    def initialize_criterion(self):
        """Initializes the loss function."""
        return torch.nn.CrossEntropyLoss()

    def initialize_optimizer_scheduler(self):
        """Initializes the optimizer and learning rate scheduler."""
        optimizer = Adam(
            self.model.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["optimizer"]["weight_decay"]
        )
        scheduler = StepLR(
            optimizer,
            step_size=self.training_config["scheduler"]["step_size"],
            gamma=self.training_config["scheduler"]["gamma"]
        )
        return optimizer, scheduler

    def initialize_early_stopping(self):
        """Initializes Early Stopping callback based on training config."""
        return EarlyStopping(
            patience=self.training_config.get("early_stopping", {}).get("patience", 5),
            min_delta=self.training_config.get("early_stopping", {}).get("min_delta", 0.001),
            verbose=True,
            save_path=self.training_config.get("early_stopping", {}).get("save_path", None)
        )

    def train(self):
        """Trains the model with early stopping and evaluates it after training."""
        start_train_time = time.time()

        for epoch in range(self.training_config["epochs"]):
            start_epoch_time = time.time()

            # Train one epoch
            train_metrics = train_one_epoch(
                model=self.model,
                data_loader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                max_grad_norm=self.training_config["gradient_clipping"]["max_grad_norm"]
            )

            # Validate model
            val_metrics = validate(
                model=self.model,
                data_loader=self.val_loader,
                criterion=self.criterion,
                device=self.device
            )

            epoch_time = time.time() - start_epoch_time
            elapsed_time = time.time() - start_train_time

            # Print training summary
            print_epoch_summary(epoch, train_metrics, val_metrics, epoch_time, elapsed_time)

            # Early Stopping Check
            if self.early_stopping(val_metrics["loss"], self.model):
                print("Early stopping triggered. Stopping training.")
                break  # Exit training loop

            self.scheduler.step()  # Step LR scheduler

        # Run final test evaluation
        return self.test()

    def test(self):
        """Evaluates the trained model on the test set."""
        print("\n# ========== Testing Model on Test Set ==========")
        test_metrics = validate(
            model=self.model,
            data_loader=self.test_loader,
            criterion=self.criterion,
            device=self.device
        )

        print(f"[Test] Loss: {test_metrics['loss']:.4f} | Accuracy: {test_metrics['accuracy']:.4f} "
              f"| Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} "
              f"| F1: {test_metrics['f1']:.4f}")

        return test_metrics
