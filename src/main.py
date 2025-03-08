import torch
from src.utils.common import parse_args, load_config
from src.dataset.domain.utils import load_dataset, split_dataset, create_dataloaders
from src.models.domain.initialize import initialize_model
from src.training.trainer import Trainer

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    config = load_config(args.config)

    # Load configurations
    dataset_config = config["dataset"]
    training_config = config["training"]
    model_config = config["model"]

    # Load dataset
    full_dataset = load_dataset(dataset_config)

    # Split dataset
    train_split, val_split, test_split = split_dataset(dataset_config, full_dataset)

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config=training_config,
        kmers_size=dataset_config["kmers_size"],
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(model_config, dataset_config, device)

    # Train and test the model
    trainer = Trainer(model, train_loader, val_loader, test_loader, training_config, device, config)
    test_metrics = trainer.train()
