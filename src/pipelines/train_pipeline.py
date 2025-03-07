import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from src.training.train import train_one_epoch
from src.training.validate import validate
from src.utils.common import print_epoch_summary
from src.training.early_stopping import EarlyStopping

def train_and_test_model(model, train_loader, val_loader, test_loader, training_config, device):
    """
    Trains the model with early stopping and evaluates it on the test set after training.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for testing data.
        training_config (dict): Training configuration.
        device (torch.device): Target device (CPU/GPU).
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(), 
        lr=training_config["learning_rate"], 
        weight_decay=training_config["optimizer"]["weight_decay"]
    )
    scheduler = StepLR(
        optimizer, 
        step_size=training_config["scheduler"]["step_size"], 
        gamma=training_config["scheduler"]["gamma"]
    )

    # Initialize Early Stopping
    early_stopping = EarlyStopping(
        patience=training_config.get("early_stopping", {}).get("patience", 5),
        min_delta=training_config.get("early_stopping", {}).get("min_delta", 0.001),
        verbose=True,
        save_path=training_config.get("early_stopping", {}).get("save_path", None)
    )

    start_train_time = time.time()

    for epoch in range(training_config["epochs"]):
        start_epoch_time = time.time()

        train_metrics = train_one_epoch(
            model=model, 
            data_loader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device, 
            max_grad_norm=training_config["gradient_clipping"]["max_grad_norm"]
        )

        val_metrics = validate(
            model=model, 
            data_loader=val_loader, 
            criterion=criterion, 
            device=device
        )

        epoch_time = time.time() - start_epoch_time
        scheduler.step()
        elapsed_time = time.time() - start_train_time

        print_epoch_summary(epoch, train_metrics, val_metrics, epoch_time, elapsed_time)

        # Early Stopping Check
        if early_stopping(val_metrics["loss"], model):
            print("Early stopping triggered. Stopping training.")
            break  # Exit training loop

    # Test after training
    print("\n# ========== Testing Model on Test Set ==========")
    test_metrics = validate(model=model, data_loader=test_loader, criterion=criterion, device=device)
    
    print(f"[Test] Loss: {test_metrics['loss']:.4f} | Accuracy: {test_metrics['accuracy']:.4f} "
          f"| Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} "
          f"| F1: {test_metrics['f1']:.4f}")

    return test_metrics
