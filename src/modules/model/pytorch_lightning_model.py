from src.modules.model.efficient_net.pytorch_lightning_efficient_net_model \
    import PyTorchLightningEfficientNetModel

class PyTorchLightningModel:
    def __new__(cls, config, experiment_execution_paths):
        if config.model_name == "EfficientNet":
            return PyTorchLightningEfficientNetModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        else:
            raise ValueError(
                f"Invalid model name: {config.model_name}. "
                f"Supported datasets are 'EfficientNet'."
            )