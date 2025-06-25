from datetime import datetime, UTC
from os.path import abspath, dirname, join
import hydra
import sys
import shap
import torch
import matplotlib.pyplot as plt

sys.path.append(abspath(join(dirname('.'), "../../")))

from src.modules.experiment_execution import setup
setup.disable_warning_messages()
setup.enforce_deterministic_behavior()
setup.set_precision(level="high")

from src.modules.data.dataloader.preprocessed_dataloader import PreprocessedDataLoader
from src.modules.data.metadataframe.metadataframe import MetadataFrame
from src.modules.experiment_execution.datetimes import ExperimentExecutionDatetimes
from src.modules.experiment_execution.config import experiment_execution_config
from src.modules.experiment_execution.info import ExperimentExecutionInfo
from src.modules.experiment_execution.prints import ExperimentExecutionPrints
from src.modules.model.model_pipeline import ModelPipeline

from src.modules.model.pytorch_lightning_model import PyTorchLightningModel
from src.modules.model.fusion_feature.resnet_fused.pytorch_lightning_resnet_fused_model \
    import PyTorchLightningResNetFusedModel
from src.modules.model.fusion_feature.resnet_fused.resnet_fused_model import ResNet_Fused_Model
import torch.nn as nn



# --- GradCAM definition ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_dict, class_idx=None):
        image_tensor = input_dict['image'].requires_grad_()
        self.model = self.model.eval()  # Ensure model is in eval mode
        output = self.model(input_dict)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[:, class_idx]
        loss.backward()

        pooled_gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (pooled_gradients * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-6)
        return cam



@hydra.main(version_base=None, config_path="../../config_files", config_name="main")
def run_explainability(config):
    print("Setting experiment ID...")
    experiment_execution_config.set_experiment_id(config)

    experiment_execution_config.delete_key(
            config, key='hyperparameter_grid_based_execution'
        )

    print("Setting experiment version ID and paths...")
    experiment_execution_config.set_experiment_version_id(config)
    experiment_execution_config.set_paths(config)
    setup.create_experiment_dir(dir_path=config.experiment_execution.paths.experiment_version_dir_path)


    print("Loading metadataframe...")
    metadataframe = MetadataFrame(config=config.data.metadataframe,
                                  experiment_execution_paths=config.experiment_execution.paths)
    print("Loading dataloader...")
    dataloader = PreprocessedDataLoader(
        config=config.data.dataloader,
        lung_nodule_metadataframe=metadataframe.get_lung_nodule_metadataframe()
    )

    print("Getting k-fold dataloaders and data names...")
    kfold_dataloaders = dataloader.get_dataloaders()
    kfold_data_names = dataloader.get_data_names()

    print(f"Setting random seed: {config.seed_value}")
    setup.set_seed(seed_value=config.seed_value)

    # Load model
    print("Loading model...")

    model_path = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\experiment_48\\version_1\\datafold_5\\models\\mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=32-var=val_auroc=0.924.ckpt"
    # Load the model with required arguments
    print(f"Model checkpoint path: {model_path}")

    checkpoint = torch.load(model_path)
    weight_dict = checkpoint['state_dict']
    
    print("Filtering state_dict keys for ResNet model...")
    weight_dict = {k: v for k, v in weight_dict.items() if k.startswith("model.resnet_model.model.")}
    print(f"Filtered state_dict keys: {list(weight_dict.keys())[:5]}... (showing first 5)")

    print("Instantiating ResNet_Fused_Model...")
    model = ResNet_Fused_Model(
        config=config.model.pytorch_lightning_model.hyperparameters,
    )

    # Sample from dataloader
    for datafold_id in range(1, 6):
        print(f"\nProcessing datafold {datafold_id}...")

        test_dataloader = kfold_dataloaders['test'][datafold_id - 1]
        print("Iterating over test dataloader...")
        for batch in test_dataloader:
            print("Preparing input dict for model...")
            input_dict = {
                'image': batch[0]['image'][0].repeat(1, 3, 1, 1),
                'lbp': batch[0]['lbp'][0].repeat(1, 3, 1, 1),
                'shape': batch[0]['shape'][0].repeat(1, 3, 1, 1),
                'fof': batch[0]['fof'][0].repeat(1, 3, 1, 1),
            }
            input_dict_train = {
                'image': batch[0]['image'].repeat(1, 3, 1, 1),
                'lbp': batch[0]['lbp'].repeat(1, 3, 1, 1),
                'shape': batch[0]['shape'].repeat(1, 3, 1, 1),
                'fof': batch[0]['fof'].repeat(1, 3, 1, 1),
            }

            model = ResNet_Fused_Model(
                config=config.model.pytorch_lightning_model.hyperparameters,
            )
            model(input_dict_train)
            model.load_state_dict(weight_dict, strict=False)
            model.eval()

            # Grad-CAM
            print("Running Grad-CAM...")
            grad_cam = GradCAM(model, target_layer=model.resnet_model.model.layer2)
            cam = grad_cam.generate_cam(input_dict)
            print("Grad-CAM completed.")


            # Plot and save
            print("Plotting and saving explainability results...")
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.imshow(batch[0]['image'][0].squeeze().cpu(), cmap='gray')
            ax.imshow(cam, cmap='jet', alpha=0.5)
            ax.set_title("Grad-CAM")
            ax.axis('off')


            plt.tight_layout()
            # Save Grad-CAM overlay
            save_path = join("C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\explainability", "explainability_sample.png")
            # Save original image
            original_image_path = join("C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\explainability", "original_image_sample.png")
            plt.imsave(original_image_path, batch[0]['image'][0].squeeze().cpu().numpy(), cmap='gray')
            plt.savefig(save_path)
            print(f"Explainability results saved to {save_path}")
            plt.close()
            break  # only one sample

    print("Execution completed.")

if __name__ == "__main__":
    run_explainability()
