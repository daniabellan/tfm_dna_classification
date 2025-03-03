import yaml
import torch


def load_yaml_config(config_file: str):
    """
    Carga un archivo YAML y lo devuelve como un diccionario.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_multiple_configs(config_files: list):
    """
    Carga múltiples archivos YAML y los combina en un único diccionario.
    """
    combined_config = {}
    for config_file in config_files:
        # Llama a load_yaml_config para cargar cada archivo y agrega su contenido
        config = load_yaml_config(config_file)
        combined_config.update(config)
    return combined_config


def load_experiment_config(experiment_file: str):
    """
    Carga la configuración completa de un experimento, incluyendo los archivos de dataset, modelo y entrenamiento.
    """
    # Cargar el archivo YAML que define los experimentos
    exp_conf = load_yaml_config(experiment_file)

    # Obtener las rutas de los archivos de configuración del experimento
    configs = exp_conf['experiment_configs']

    # Cargar todas las configuraciones necesarias
    dataset_config_files = configs["dataset"]
    model_config_files = configs["model"]
    train_config_files = configs["train"]

    # Cargar todos los archivos de configuración en diccionarios separados
    dataset_config = load_multiple_configs(dataset_config_files)
    model_config = load_multiple_configs(model_config_files)
    train_config = load_multiple_configs(train_config_files)

    # Experiment name
    exp_name = exp_conf["experiment_name"]

    return [exp_name, dataset_config, model_config, train_config]


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def print_epoch_summary(epoch, train_metrics, val_metrics, epoch_time, elapsed_time):
    """Imprime el resumen de cada época y los tiempos"""
    print(f"\n# ========== Train Epoch {epoch+1}/100 ==========")
    print(f"[Train] Loss: {train_metrics['loss']:.4f} | Accuracy: {train_metrics['accuracy']:.4f} | Precision: {train_metrics['precision']:.4f} | Recall: {train_metrics['recall']:.4f} | F1: {train_metrics['f1']:.4f}")
    print(f"[Val]   Loss: {val_metrics['loss']:.4f} | Accuracy: {val_metrics['accuracy']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
    print(f"Epoch Time: {epoch_time:.4f} seconds | Elapsed Time: {elapsed_time:.4f} seconds")
