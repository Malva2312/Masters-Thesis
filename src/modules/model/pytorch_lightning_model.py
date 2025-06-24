from src.modules.model.standalone.effnet.pytorch_lightning_efficient_net_model \
    import PyTorchLightningEfficientNetModel

from src.modules.model.standalone.linear_svm.pytorch_lightning_linear_svm_model \
    import PyTorchLightningLinearSVMModel

from src.modules.model.standalone.resnet.pytorch_lightning_resnet_model \
    import PyTorchLightningResNetModel

from src.modules.model.fusion_feature.resnet_fused.pytorch_lightning_resnet_fused_model \
    import PyTorchLightningResNetFusedModel

from src.modules.model.fusion_feature.effnet_fused.pytorch_lightning_effnet_fused_model \
    import PyTorchLightningEffNetFusedModel

from src.modules.model.fusion_feature.convnext.pytorch_lightning_convnext_fused_model \
    import PyTorchLightningConvNextFusedModel

    
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
        elif config.model_name == "ResNet":  # <-- Add this block
            return PyTorchLightningResNetModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        elif config.model_name == "ResNetFusionModel":
            return PyTorchLightningResNetFusedModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        elif config.model_name == "EffNetFusionModel":
            return PyTorchLightningEffNetFusedModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        elif config.model_name == "ConvNextFusionModel":
            return PyTorchLightningConvNextFusedModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        else:
            raise ValueError(
                f"Invalid model name: {config.model_name}. "
                f"Supported datasets are 'EfficientNet', 'LinearSVM', 'EfficientNet_LBP', 'EfficientNet_SVM', 'ResNet'."
            )