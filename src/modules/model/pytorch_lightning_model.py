from src.modules.model.standalone.effnet.pytorch_lightning_efficient_net_model \
    import PyTorchLightningEfficientNetModel

from src.modules.model.standalone.linear_svm.pytorch_lightning_linear_svm_model \
    import PyTorchLightningLinearSVMModel

from src.modules.model.fusion_feature.effnet_lbp.pytorch_lightning_effnet_lbp_model \
    import PyTorchLightningEfficientNetLBPModel

from src.modules.model.fusion_decision.effnet_svm.pytorch_lightning_effnet_svm_fusion_model \
    import PyTorchLightningEffNetSVMFusionModel

from src.modules.model.standalone.resnet.pytorch_lightning_resnet_model \
    import PyTorchLightningResNetModel

from src.modules.model.standalone.vgg.pytorch_lightning_vgg_model \
    import PyTorchLightningVGGModel

from src.modules.model.standalone.alexnet.pytorch_lightning_alexnet_model \
    import PyTorchLightningAlexNetModel

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
        elif config.model_name == "ResNet":  # <-- Add this block
            return PyTorchLightningResNetModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        elif config.model_name == "VGG":
            return PyTorchLightningVGGModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        elif config.model_name == "AlexNet":
            return PyTorchLightningAlexNetModel(
                config=config.hyperparameters,
                experiment_execution_paths=experiment_execution_paths
            )
        else:
            raise ValueError(
                f"Invalid model name: {config.model_name}. "
                f"Supported datasets are 'EfficientNet', 'LinearSVM', 'EfficientNet_LBP', 'EfficientNet_SVM', 'ResNet'."
            )