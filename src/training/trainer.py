import time
import torch
import seaborn as sns
import tempfile
import mlflow
import markdown
import mlflow.pytorch
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from src.utils.logging_config import logger, get_log_file
from src.training.train import train_one_epoch
from src.training.validate import validate
from src.training.early_stopping import EarlyStopping

class Trainer:
    """
    Trainer class to handle model training, validation, and logging with MLflow.
    """
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader, 
                 test_loader, 
                 training_config, 
                 device, 
                 config, 
                 logging_config, 
                 dataset_config, 
                 model_config):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.training_config = training_config
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.logging_config = logging_config
        self.device = device
        self.config = config  # Store full config to log

        logger.info("Training started with configuration:")
        logger.info(f"Training: {self.training_config}")
        logger.info(f"Dataset: {self.dataset_config}")
        logger.info(f"Model: {self.model_config}")
        logger.info(f"Logging: {self.logging_config}")
        logger.info(f"Description: {config['description']}")

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = Adam(
            model.parameters(), 
            lr=training_config["learning_rate"], 
            weight_decay=training_config["optimizer"]["weight_decay"]
        )
        self.scheduler = StepLR(
            self.optimizer, 
            step_size=training_config["scheduler"]["step_size"], 
            gamma=training_config["scheduler"]["gamma"]
        )

        self.early_stopping = EarlyStopping(
            patience=training_config.get("early_stopping", {}).get("patience", 5),
            min_delta=training_config.get("early_stopping", {}).get("min_delta", 0.001),
            verbose=True,
            save_path=training_config.get("early_stopping", {}).get("save_path", None)
        )

        # Set MLflow tracking URI 
        mlflow_tracking_uri = logging_config.get("mlflow_tracking_uri", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # Create run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.run_name = f"{timestamp}_samples{dataset_config['num_samples']}"
        self.experiment_name = config["name"]

        # Enable system metrics logging
        mlflow.enable_system_metrics_logging()

    def train(self):
        """
        Trains the model with MLflow logging and early stopping.
        """
        # Configure MLFlow tracking 
        mlflow.set_experiment(self.experiment_name)
        
        tags = {
            'mlflow.note.content': markdown.markdown(self.config["description"]),
            'mlflow.user': "Daniel Abellan Sanchez"
        }

        with mlflow.start_run(run_name=self.run_name, tags=tags):
            # Log configuration parameters
            mlflow.log_params(self.config["model"])
            mlflow.log_params(self.config["training"])
            mlflow.log_params(self.config["dataset"])

            start_train_time = time.time()
            logger.info("Training started!")

            for epoch in range(self.training_config["epochs"]):
                start_epoch_time = time.time()

                train_metrics = train_one_epoch(
                    model=self.model, 
                    data_loader=self.train_loader, 
                    criterion=self.criterion, 
                    optimizer=self.optimizer, 
                    device=self.device, 
                    max_grad_norm=self.training_config["gradient_clipping"]["max_grad_norm"],
                )

                val_metrics = validate(
                    model=self.model, 
                    data_loader=self.val_loader, 
                    criterion=self.criterion, 
                    device=self.device
                )

                epoch_time = time.time() - start_epoch_time
                elapsed_time = time.time() - start_train_time

                print(f"\nEpoch time: {epoch_time:.4f} sec")
                print(f"Elapsed time: {elapsed_time:.4f} sec\n")

                logger.info(f"Epoch {epoch+1}/{self.training_config['epochs']} completed in {epoch_time:.4f} sec.")
                logger.info(f"Total elapsed time: {elapsed_time:.4f} sec.")
                
                mlflow.log_artifact(str(get_log_file()))
                
                # Log train & validation metrics
                mlflow.log_metrics(train_metrics, step=epoch)
                mlflow.log_metrics(val_metrics, step=epoch)

                # Check early stopping
                if self.early_stopping(val_metrics["val_loss"], self.model):
                    logger.warning("Early stopping triggered. Training stopped.")
                    print("Early stopping triggered. Stopping training.")
                    break

                self.scheduler.step()

            # Log final test metrics and confusion matrix
            return self.test()

    def test(self):
        """
        Evaluates the trained model on the test set and logs test results & confusion matrix.
        """
        logger.info("Running final evaluation on the test set...")
        test_metrics, _, _, conf_matrix = self.evaluate_model_on_test()

        # Log All Metrics in MLflow
        mlflow.log_metrics(test_metrics)

        # Log Confusion Matrix
        self.log_confusion_matrix(conf_matrix)

        # Save the trained model as an artifact
        # Get data sample of test_loader
        sample_signals, sample_sequences, _ = next(iter(self.test_loader))

        # Move sample date to model device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_signals = sample_signals.to(device)
        sample_sequences = sample_sequences.to(device)

        # Inference model
        logger.success(f"Saving model...")
        mlflow.log_metrics(test_metrics)

        self.model.eval()  
        with torch.no_grad():
            predictions = self.model(sample_signals, sample_sequences)

        # Convert input to numpy
        sample_signals_np = sample_signals.cpu().numpy()
        sample_sequences_np = sample_sequences.cpu().numpy()
        predictions_np = predictions.cpu().numpy()

        # Convert inputs to lists
        model_input = {
            "signals": sample_signals_np.tolist(),  
            "sequences": sample_sequences_np.tolist() 
        }

        # Create signature using model input and output 
        signature = mlflow.models.infer_signature(
            model_input=model_input,  
            model_output=predictions_np.tolist() 
        )

        # Log model to MLFlow
        mlflow.pytorch.log_model(pytorch_model = self.model, 
                                 artifact_path="model", 
                                 signature=signature)
        logger.success(f"Model saved in MLFlow!")
        
        mlflow.log_artifact(str(get_log_file()))
        return test_metrics


    def evaluate_model_on_test(self):
        """
        Runs model on test set and returns a detailed set of evaluation metrics.
        """
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for signals, sequences, labels in self.test_loader:
                signals, sequences, labels = signals.to(self.device), sequences.to(self.device), labels.to(self.device)

                outputs = self.model(signals, sequences)
                _, preds = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Compute Multi-Class Metrics with zero_division=0
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Aggregate Metrics Dictionary
        test_metrics = {
            "test_accuracy": accuracy,
            "test_precision_macro": precision_macro,
            "test_recall_macro": recall_macro,
            "test_f1_macro": f1_macro
        }

        # Print Test Summary
        print(f"\n# ========== Test Set Evaluation ==========")
        print(f"[Test] Accuracy: {test_metrics['test_accuracy']:.4f} | "
              f"Precision: {test_metrics['test_precision_macro']:.4f} | "
              f"Recall: {test_metrics['test_recall_macro']:.4f} | "
              f"F1: {test_metrics['test_f1_macro']:.4f}")

        logger.success(f"[Test] Accuracy: {test_metrics['test_accuracy']:.4f} | "
              f"Precision: {test_metrics['test_precision_macro']:.4f} | "
              f"Recall: {test_metrics['test_recall_macro']:.4f} | "
              f"F1: {test_metrics['test_f1_macro']:.4f}")
        
        logger.info(f"\n{conf_matrix}")
        mlflow.log_artifact(str(get_log_file()))
        
        return test_metrics, y_true, y_pred, conf_matrix


    def log_confusion_matrix(self, conf_matrix):
        """
        Saves the confusion matrix and logs it as an MLflow artifact.
        """
        logger.info("Logging confusion matrix...")
    
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(conf_matrix.shape[0]), yticklabels=np.arange(conf_matrix.shape[0]))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        with tempfile.TemporaryDirectory() as tmp_dir:
            conf_matrix_path = Path(tmp_dir) / "conf_matrix.png"
            
            # Save image in temporal path
            fig.savefig(conf_matrix_path)
            plt.close(fig)  

            # Log as MLflow artifact
            mlflow.log_artifact(str(conf_matrix_path))

        logger.success(f"Confusion matrix saved as MLflow artifact: {conf_matrix_path}")
        print(f"Confusion matrix saved as artifact: {conf_matrix_path}")
