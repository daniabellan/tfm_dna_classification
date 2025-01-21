import time
import mlflow
from datetime import datetime

class MLFlowLogger():
    def __init__(self, 
                 experiment_config:dict, 
                 model_config:dict, 
                 experiment_name:str, 
                 len_train_dataset:int, 
                 len_val_dataset:int, 
                 len_test_dataset:int):
        self.experiment_name = experiment_name
        self.train_size = len_train_dataset
        self.val_size = len_val_dataset
        self.test_size = len_test_dataset

        # Ordenar configuracion modelo y experimento
        sorted_model_config = {key: value for key, value in sorted(model_config.items())}
        sorted_experiment_config = {key: value for key, value in sorted(experiment_config.items())}
        
        # Usamos la fecha y hora en formato ISO 8601
        self.run_name = "run_" + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        mlflow.set_tracking_uri(sorted_experiment_config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)

        # Log de los tamaños de los datasets
        mlflow.log_param("train_size", self.train_size)
        mlflow.log_param("val_size", self.val_size)
        mlflow.log_param("test_size", self.test_size)

        # Log de los parametros del experimento
        self.setup_params(config = sorted_experiment_config, 
                          config_type = "exp")

        # Log de los parametros del modelo
        self.setup_params(config = sorted_model_config,
                          config_type = "model")

    def setup_params(self, config:dict, config_type:str):
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    mlflow.log_param(f"{config_type}_{key}_{sub_key}", sub_value)
            else:
                mlflow.log_param(f"{config_type}_{key}", value)

    def log_metrics(self, metrics, epoch):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=epoch)
    
    def log_epoch_time(self, epoch_time):
        mlflow.log_metric("epoch_time", epoch_time)

    def start_timer(self):
        self.start_time = time.time()

    def end_mlflow(self):
        mlflow.end_run()
