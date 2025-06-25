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

# --- SHAP wrapper ---
class SHAPFeatureWrapper(torch.nn.Module):
    def __init__(self, model, fixed_image_tensor):
        super().__init__()
        self.model = model.eval()
        self.fixed_image = fixed_image_tensor

    def forward(self, features):
        B = features.shape[0]
        images = self.fixed_image.repeat(B, 1, 1, 1)
        input_dict = {
            'image': images,
            'features': features
        }
        return self.model(input_dict)

@hydra.main(version_base=None, config_path="../../config_files", config_name="main")
def run_explainability(config):
    experiment_execution_config.set_experiment_version_id(config)
    experiment_execution_config.set_paths(config)
    setup.create_experiment_dir(dir_path=config.experiment_execution.paths.experiment_version_dir_path)

    experiment_execution_datetimes = ExperimentExecutionDatetimes(
        experiment_execution_paths=config.experiment_execution.paths
    )
    experiment_execution_datetimes.add_event("overall_execution", "start", str(datetime.now(UTC).replace(microsecond=0)))

    metadataframe = MetadataFrame(config=config.data.metadataframe,
                                  experiment_execution_paths=config.experiment_execution.paths)
    dataloader = PreprocessedDataLoader(
        config=config.data.dataloader,
        lung_nodule_metadataframe=metadataframe.get_lung_nodule_metadataframe()
    )

    # Load model
    model_path = "/nas-ctm01/homes/jmalva/Masters-Thesis/experiment_results/experiment_46/version_1/datafold_1/models/mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=29-var=last_epoch.ckpt"
    model = torch.load(model_path)
    model.eval()

    # Sample from dataloader
    for batch in dataloader.test_dataloader():
        input_dict = {
            'image': batch['image'],
            'features': batch['features']
        }

        # Grad-CAM
        grad_cam = GradCAM(model, target_layer=model.resnet_model.model.layer4[-1])
        cam = grad_cam.generate_cam(input_dict)

        # SHAP
        wrapped = SHAPFeatureWrapper(model, batch['image'])
        background = batch['features'][:30]
        test_sample = batch['features'][0:1]
        explainer = shap.KernelExplainer(lambda x: wrapped(torch.tensor(x).float()).detach().numpy(), background.numpy())
        shap_values = explainer.shap_values(test_sample.numpy())

        # Plot and save
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(batch['image'][0].squeeze().cpu(), cmap='gray')
        ax[0].imshow(cam, cmap='jet', alpha=0.5)
        ax[0].set_title("Grad-CAM")
        ax[0].axis('off')

        ax[1].barh(range(len(shap_values[0][0])), shap_values[0][0], color='purple')
        ax[1].set_title("SHAP (Features)")
        ax[1].set_xlabel("SHAP Value")

        plt.tight_layout()
        plt.savefig(join("/nas-ctm01/homes/jmalva/Masters-Thesis/data/explainability", "explainability_sample.png"))
        plt.close()
        break  # only one sample

    experiment_execution_datetimes.add_event("overall_execution", "end", str(datetime.now(UTC).replace(microsecond=0)))
    experiment_execution_datetimes.save()

if __name__ == "__main__":
    run_explainability()
