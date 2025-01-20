import time
import mlflow
from datetime import datetime

class MLFlowLogger():
    def __init__(self, config: dict, experiment_name: str):
        self.experiment_name = experiment_name
        self.config = config
        self.run_name = "run_" + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)

        self.setup_params()

        # Inicializamos los tiempos
        self.start_time = None
        self.epoch_times = []

    def setup_params(self):
        for key, value in self.config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    mlflow.log_param(f"{key}_{sub_key}", sub_value)
            else:
                mlflow.log_param(key, value)

    def start_timer(self):
        """Inicia el temporizador para el entrenamiento"""
        self.start_time = time.time()

    def log_epoch_time(self, epoch_time):
        """Guarda el tiempo de cada época"""
        self.epoch_times.append(epoch_time)

    def log_metrics(self, metrics, epoch):
        """Registra las métricas de entrenamiento o validación"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=epoch)

    def log_training_times(self):
        """Registra el tiempo total de entrenamiento y el tiempo medio por época"""
        total_time = time.time() - self.start_time
        average_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0

        # Registrar los tiempos en MLFlow
        mlflow.log_metric("total_training_time", total_time)
        mlflow.log_metric("average_epoch_time", average_epoch_time)

    def end_mlflow(self):
        """Finaliza el registro en MLFlow"""
        self.log_training_times()  # Registrar los tiempos al finalizar el entrenamiento
        mlflow.end_run()
