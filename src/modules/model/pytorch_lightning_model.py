from src.modules.model.efficient_net.pytorch_lightning_efficient_net_model \
    import PyTorchLightningEfficientNetModel

from src.modules.model.linear_svm.pytorch_lightning_linear_svm_model \
    import PyTorchLightningLinearSVMModel

from src.modules.model.efficient_net_lbp.pytorch_lightning_effnet_lbp_model \
    import PyTorchLightningEfficientNetLBPModel

from src.modules.model.efficient_net_svm.pytorch_lightning_effnet_svm_fusion_model \
    import PyTorchLightningEffNetSVMFusionModel


class PyTorchLightningModel:
    def __new__(cls, config, experiment_execution_paths):
        if config.model_name == "EfficientNet":
            return PyTorchLightningEfficientNetModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        elif config.model_name == "LinearSVM":
            return PyTorchLightningLinearSVMModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        elif config.model_name == "EfficientNet_LBP":
            return PyTorchLightningEfficientNetLBPModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        elif config.model_name == "EfficientNet_SVM":
            return PyTorchLightningEffNetSVMFusionModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        else:
            raise ValueError(
                f"Invalid model name: {config.model_name}. "
                f"Supported datasets are 'EfficientNet'."
            )