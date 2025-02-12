import time
import json
import mlflow
from datetime import datetime

class MLFlowLogger():
    def __init__(self, 
                 train_config: dict,
                 model_config: dict, 
                 dataset_config: dict,
                 experiment_name: str, 
                 len_train_dataset: int, 
                 len_val_dataset: int, 
                 len_test_dataset: int):

        self.experiment_name = experiment_name
        self.train_size = len_train_dataset
        self.val_size = len_val_dataset
        self.test_size = len_test_dataset

        # Ordenar configuraciones para consistencia en los logs
        sorted_model_config = dict(sorted(model_config.items()))
        sorted_train_config = dict(sorted(train_config.items()))
        sorted_dataset_config = dict(sorted(dataset_config.items()))

        # Nombre único de la ejecución con timestamp
        self.run_name = "run_" + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        # Configuración de MLFlow
        mlflow.set_tracking_uri(sorted_train_config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)

        # Log de tamaños de los datasets
        mlflow.log_param("train_size", self.train_size)
        mlflow.log_param("val_size", self.val_size)
        mlflow.log_param("test_size", self.test_size)

        # Log de configuración
        self.setup_params(sorted_train_config, "train")
        self.setup_params(sorted_model_config, "model")
        self.setup_params(sorted_dataset_config, "data")

    def setup_params(self, config: dict, prefix: str):
        """ Registra parámetros en MLflow con prefijos organizados """
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    mlflow.log_param(f"{prefix}_{key}_{sub_key}", sub_value)
            elif isinstance(value, list):  
                if key == "real_dataset":  # Si la clave es "real_dataset", aplicamos el formato especial
                    formatted_data = [{"label": i, "data": path} for i, path in enumerate(value)]
                    mlflow.log_param(f"{prefix}_{key}", json.dumps(formatted_data))  # Guardar como JSON string
                else:
                    mlflow.log_param(f"{prefix}_{key}", str(value))  # Convertir listas normales a strings
            else:
                mlflow.log_param(f"{prefix}_{key}", value)

    def log_metrics(self, metrics: dict, epoch: int):
        """ Registra métricas de una epoch en MLflow """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=epoch)

    def log_epoch_time(self, epoch_time: float):
        """ Registra el tiempo por epoch """
        mlflow.log_metric("epoch_time", epoch_time)

    def start_timer(self):
        """ Inicia el temporizador para medir el tiempo de ejecución """
        self.start_time = time.time()

    def end_mlflow(self):
        """ Finaliza la ejecución en MLflow """
        mlflow.end_run()
