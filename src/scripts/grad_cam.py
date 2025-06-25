from datetime import datetime
from os.path import abspath, dirname, join
import os
import hydra
import sys
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

sys.path.append(abspath(join(dirname('.'), "../../")))

from src.modules.experiment_execution import setup
setup.disable_warning_messages()
setup.enforce_deterministic_behavior()
setup.set_precision(level="high")

from src.modules.data.dataloader.preprocessed_dataloader import PreprocessedDataLoader
from src.modules.data.metadataframe.metadataframe import MetadataFrame
from src.modules.experiment_execution.config import experiment_execution_config
from src.modules.model.fusion_feature.resnet_fused.resnet_fused_model import ResNet_Fused_Model
from skimage.transform import resize


fused_model_ckpt = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\experiment_48\\version_1\\datafold_5\\models\\mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=32-var=val_auroc=0.924.ckpt"
non_fused_model_ckpt = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\experiment_46\\version_1\\datafold_3\\models\\mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=70-var=last_epoch.ckpt"
datafold_idx = [4]

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
        self.model.eval()
        output = self.model(input_dict)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[:, class_idx].backward()

        pooled_gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (pooled_gradients * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-6)
        return cam

def save_images(base_path, name_prefix, image_tensor, cam_array):
    os.makedirs(base_path, exist_ok=True)

    original_path = join(base_path, f"{name_prefix}_original.png")
    zoom_path = join(base_path, f"{name_prefix}_zoom.png")
    cam_overlay_path = join(base_path, f"{name_prefix}_gradcam.png")

    original_np = image_tensor.squeeze().cpu().numpy()
    plt.imsave(original_path, original_np, cmap='gray')

    # Zoom: resize the whole image to 512x512 for easier interpretation
    zoomed_resized = resize(original_np, (512, 512), order=0, mode='reflect', anti_aliasing=False, preserve_range=True)
    plt.imsave(zoom_path, zoomed_resized.astype(original_np.dtype), cmap='gray')

    #original_np = (original_np - original_np.min()) / (original_np.max() - original_np.min())

    # CAM overlay
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(original_np, cmap='gray')
    ax.imshow(cam_array, cmap='jet', alpha=0.5)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(cam_overlay_path, dpi=300)
    plt.close()

@hydra.main(version_base=None, config_path="../../config_files", config_name="main")
def run_explainability(config):
    print("Setting seed:", config.seed_value)
    setup.set_seed(config.seed_value)

    experiment_execution_config.set_experiment_id(config)
    experiment_execution_config.delete_key(config, key='hyperparameter_grid_based_execution')
    experiment_execution_config.set_experiment_version_id(config)
    experiment_execution_config.set_paths(config)
    setup.create_experiment_dir(config.experiment_execution.paths.experiment_version_dir_path)

    metadataframe = MetadataFrame(config=config.data.metadataframe,
                                  experiment_execution_paths=config.experiment_execution.paths)
    dataloader = PreprocessedDataLoader(
        config=config.data.dataloader,
        lung_nodule_metadataframe=metadataframe.get_lung_nodule_metadataframe()
    )
    kfold_dataloaders = dataloader.get_dataloaders()


    # Load checkpoints
    fused_ckpt = torch.load(fused_model_ckpt)
    non_fused_ckpt = torch.load(non_fused_model_ckpt)

    fused_weights = {k: v for k, v in fused_ckpt['state_dict'].items()}
    non_fused_weights = {k: v for k, v in non_fused_ckpt['state_dict'].items()}

    # Process only the first fold
    for fold_idx, test_dataloader in enumerate(kfold_dataloaders['test']):
        if datafold_idx:
            if fold_idx not in datafold_idx:
                continue
        for idx, batch in enumerate(test_dataloader):
            for img_idx in range(batch[0]['image'].shape[0]):
                input_dict = {
                    'image': batch[0]['image'][img_idx].unsqueeze(0).repeat(1, 3, 1, 1),
                    'lbp': batch[0]['lbp'][img_idx].unsqueeze(0).repeat(1, 3, 1, 1),
                    'shape': batch[0]['shape'][img_idx].unsqueeze(0).repeat(1, 3, 1, 1),
                    'fof': batch[0]['fof'][img_idx].unsqueeze(0).repeat(1, 3, 1, 1),
                }

                input_dict_train = {
                    'image': batch[0]['image'].repeat(1, 3, 1, 1),
                    'lbp': batch[0]['lbp'].repeat(1, 3, 1, 1),
                    'shape': batch[0]['shape'].repeat(1, 3, 1, 1),
                    'fof': batch[0]['fof'].repeat(1, 3, 1, 1),
                }

                # Fused model
                config.model.pytorch_lightning_model.hyperparameters.resnet_config = config.model.pytorch_lightning_model.hyperparameters.fused_resnet_config
                fused_model = ResNet_Fused_Model(config=config.model.pytorch_lightning_model.hyperparameters)
                fused_model(input_dict_train)
                fused_model.load_state_dict(fused_weights, strict=False)
                fused_model.eval()

                cam_fused = GradCAM(fused_model, fused_model.resnet_model.model.layer2).generate_cam(input_dict)

                # Non-fused model
                config.model.pytorch_lightning_model.hyperparameters.resnet_config = config.model.pytorch_lightning_model.hyperparameters.base_resnet_config
                non_fused_model = ResNet_Fused_Model(config=config.model.pytorch_lightning_model.hyperparameters)
                non_fused_model(input_dict_train)  # trigger lazy init
                non_fused_model.load_state_dict(non_fused_weights, strict=False)
                non_fused_model.eval()

                cam_non_fused = GradCAM(non_fused_model, non_fused_model.resnet_model.model.layer2).generate_cam(input_dict)

                # Save in separate folder for each image
                base_dir = join(config.experiment_execution.paths.experiment_dir_path, "explainability")
                fused_dir = join(base_dir, "fused", f"fold_{fold_idx}", f"batch_{idx}_{img_idx}")
                non_fused_dir = join(base_dir, "non_fused", f"fold_{fold_idx}", f"batch_{idx}_{img_idx}")
                save_images(fused_dir, "sample", batch[0]['image'][img_idx], cam_fused)
                save_images(non_fused_dir, "sample", batch[0]['image'][img_idx], cam_non_fused)
                print(f"Saved fused and non-fused explainability outputs for fold {fold_idx}, batch {idx}, image {img_idx}.")

if __name__ == "__main__":
    run_explainability()
