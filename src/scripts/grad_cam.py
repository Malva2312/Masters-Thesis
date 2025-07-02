from datetime import datetime
from os.path import abspath, dirname, join
import os
import hydra
import sys
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
import json
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np

sys.path.append(abspath(join(dirname('.'), "../../")))

from src.modules.experiment_execution import setup
setup.disable_warning_messages()
setup.enforce_deterministic_behavior()
setup.set_precision(level="high")

from src.modules.data.dataloader.preprocessed_dataloader import PreprocessedDataLoader
from src.modules.data.metadataframe.metadataframe import MetadataFrame
from src.modules.experiment_execution.config import experiment_execution_config
from src.modules.model.fusion_feature.resnet_fused.resnet_fused_model import ResNet_Fused_Model

fused_model_ckpt = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\experiment_48\\version_1\\datafold_5\\models\\mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=32-var=val_auroc=0.924.ckpt"
non_fused_model_ckpt = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\experiment_46\\version_1\\datafold_5\\models\\mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=39-var=last_epoch.ckpt"
datafold_idx = [4]

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
        self.target_layer.register_full_backward_hook(backward_hook)

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

class GradCAMPlusPlus:
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
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_dict, class_idx=None):
        image_tensor = input_dict['image'].requires_grad_()
        self.model.eval()
        output = self.model(input_dict)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients  # shape: (B, C, H, W)
        activations = self.activations  # shape: (B, C, H, W)

        # Grad-CAM++ weights calculation
        gradients = gradients[0]
        activations = activations[0]
        grads2 = gradients ** 2
        grads3 = grads2 * gradients

        # global sum over spatial dimensions
        sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True)

        eps = 1e-8
        alpha_num = grads2
        alpha_denom = 2 * grads2 + sum_activations * grads3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom) * eps)
        alphas = alpha_num / alpha_denom

        positive_gradients = torch.relu(score.exp() * gradients)
        weights = (alphas * positive_gradients).sum(dim=(1, 2))

        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=image_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        cam = cam.squeeze().cpu().detach().numpy()
        denom = cam.max() - cam.min()
        cam = (cam - cam.min()) / (denom + 1e-6) if denom > 0 else cam
        return cam


class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        self.target_layer.register_forward_hook(forward_hook)

    def generate_cam(self, input_dict, class_idx=None, n_samples=32, batch_size=16):
        """
        Args:
            input_dict: dictionary with {'image': image_tensor}
            class_idx: target class for visualization
            n_samples: how many masks to use (default 32)
            batch_size: mini-batch size for masked forward passes
        """
        image_tensor = input_dict['image']
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_dict)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        # Forward to get activations
        _ = self.model(input_dict)
        activations = self.activations[0]  # (C, H, W)
        b, c, h, w = self.activations.shape

        # Normalize activations to [0, 1]
        activation_maps = (activations - activations.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0])
        activation_maps = activation_maps / (activation_maps.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] + 1e-8)

        weights = []
        for i in range(c):
            # Upsample activation to input size
            act = activation_maps[i].unsqueeze(0).unsqueeze(0)
            upsampled = torch.nn.functional.interpolate(act, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
            norm_mask = upsampled.squeeze().clamp(0, 1)
            # Apply mask to input image
            masked_img = image_tensor[0] * norm_mask
            masked_input = {'image': masked_img.unsqueeze(0)}
            with torch.no_grad():
                score = self.model(masked_input)[0, class_idx].item()
            weights.append(score)
        weights = torch.tensor(weights)
        # Weighted sum
        cam = (weights.view(-1, 1, 1) * activation_maps).sum(dim=0)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                              size=image_tensor.shape[2:],
                                              mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-6)
        return cam

class SmoothGradCAMPlusPlus:
    def __init__(self, model, target_layer, n_samples=25, stdev_spread=0.15):
        self.model = model.eval()
        self.target_layer = target_layer
        self.n_samples = n_samples
        self.stdev_spread = stdev_spread
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_dict, class_idx=None):
        image_tensor = input_dict['image']
        self.model.eval()
        stdev = self.stdev_spread * (image_tensor.max() - image_tensor.min()).item()
        cams = []
        for i in range(self.n_samples):
            noise = torch.normal(0, stdev, size=image_tensor.shape, device=image_tensor.device)
            noisy_image = image_tensor + noise
            noisy_input_dict = {'image': noisy_image}
            # Forward
            output = self.model(noisy_input_dict)
            if class_idx is None:
                idx = output.argmax(dim=1).item()
            else:
                idx = class_idx
            self.model.zero_grad()
            score = output[:, idx]
            score.backward(retain_graph=True)
            gradients = self.gradients[0]
            activations = self.activations[0]
            grads2 = gradients ** 2
            grads3 = grads2 * gradients
            sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True)
            eps = 1e-8
            alpha_num = grads2
            alpha_denom = 2 * grads2 + sum_activations * grads3
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom) * eps)
            alphas = alpha_num / alpha_denom
            positive_gradients = torch.relu(score.exp() * gradients)
            weights = (alphas * positive_gradients).sum(dim=(1, 2))
            cam = (weights[:, None, None] * activations).sum(dim=0)
            cam = torch.relu(cam)
            cam = torch.nn.functional.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=image_tensor.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            cam = cam.squeeze().cpu().detach().numpy()
            denom = cam.max() - cam.min()
            cam = (cam - cam.min()) / (denom + 1e-6) if denom > 0 else cam
            cams.append(cam)
        # Average the cams
        cam = np.mean(np.stack(cams), axis=0)
        cam = (cam - cam.min()) / (cam.max() + 1e-6)
        return cam
   
def save_images(base_path, name_prefix, image_tensor, cam_array):
    os.makedirs(base_path, exist_ok=True)
    original_path = join(base_path, f"{name_prefix}_original.png")
    zoom_path = join(base_path, f"{name_prefix}_zoom.png")
    cam_overlay_path = join(base_path, f"{name_prefix}_gradcam.png")
    original_np = image_tensor.squeeze().cpu().numpy()
    plt.imsave(original_path, original_np, cmap='gray')
    zoomed_resized = resize(original_np, (512, 512), order=0, mode='reflect', anti_aliasing=False, preserve_range=True)
    plt.imsave(zoom_path, zoomed_resized.astype(original_np.dtype), cmap='gray')
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

    fused_ckpt = torch.load(fused_model_ckpt)
    non_fused_ckpt = torch.load(non_fused_model_ckpt)
    fused_weights = {k.replace("model.", "", 1): v for k, v in fused_ckpt['state_dict'].items()}
    non_fused_weights = {k.replace("model.", "", 1): v for k, v in non_fused_ckpt['state_dict'].items()}

    all_labels = []
    all_fused_preds = []
    all_non_fused_preds = []

    for fold_idx, test_dataloader in enumerate(kfold_dataloaders['test']):
        if datafold_idx and fold_idx not in datafold_idx:
            continue

        iterator = iter(test_dataloader)
        first_batch = next(iterator)
        
        input_dict_train = {
            'image': first_batch[0]['image'].repeat(1, 1, 1, 1),
            'lbp': first_batch[0]['lbp'].repeat(1, 1, 1, 1),
            'shape': first_batch[0]['shape'].repeat(1, 1, 1, 1),
            'fof': first_batch[0]['fof'].repeat(1, 1, 1, 1),
        }

        # Fused model
        config.model.pytorch_lightning_model.hyperparameters.resnet_config = config.model.pytorch_lightning_model.hyperparameters.fused_resnet_config
        fused_model = ResNet_Fused_Model(config=config.model.pytorch_lightning_model.hyperparameters)
        fused_model(input_dict_train)
        fused_model.load_state_dict(fused_weights, strict=True)
        fused_model.eval()
        fused_model.resnet_model.eval()
        fused_model.resnet_model.model.eval()
        
        # Non-fused model
        config.model.pytorch_lightning_model.hyperparameters.resnet_config = config.model.pytorch_lightning_model.hyperparameters.base_resnet_config
        non_fused_model = ResNet_Fused_Model(config=config.model.pytorch_lightning_model.hyperparameters)
        non_fused_model.load_state_dict(non_fused_weights, strict=True)
        non_fused_model.eval()
        non_fused_model.resnet_model.eval()
        non_fused_model.resnet_model.model.eval()

        for idx, batch in enumerate(test_dataloader):
            batch_size = batch[0]['image'].shape[0]
            labels = batch[1]

            for img_idx in range(batch_size):
                input_dict = {
                    'image': batch[0]['image'][img_idx].unsqueeze(0).repeat(1, 1, 1, 1),
                    'lbp': batch[0]['lbp'][img_idx].unsqueeze(0).repeat(1, 1, 1, 1),
                    'shape': batch[0]['shape'][img_idx].unsqueeze(0).repeat(1, 1, 1, 1),
                    'fof': batch[0]['fof'][img_idx].unsqueeze(0).repeat(1, 1, 1, 1),
                }

                file_base = batch[2][img_idx]

                cam_fused = GradCAM(fused_model, fused_model.resnet_model.model.layer3).generate_cam(input_dict)
                cam_non_fused = GradCAM(non_fused_model, non_fused_model.resnet_model.model.layer3).generate_cam(input_dict)

                #cam_fused = GradCAMPlusPlus(fused_model, fused_model.resnet_model.model.layer3[-1]).generate_cam(input_dict)
                #cam_non_fused = GradCAMPlusPlus(non_fused_model, non_fused_model.resnet_model.model.layer3[-1]).generate_cam(input_dict)

                #cam_fused = SmoothGradCAMPlusPlus(fused_model, fused_model.resnet_model.model.layer3).generate_cam(input_dict)
                #cam_non_fused = SmoothGradCAMPlusPlus(non_fused_model, non_fused_model.resnet_model.model.layer3).generate_cam(input_dict)

                #cam_fused = ScoreCAM(fused_model, fused_model.resnet_model.model.layer3).generate_cam(input_dict)
                #cam_non_fused = ScoreCAM(non_fused_model, non_fused_model.resnet_model.model.layer3).generate_cam(input_dict)
                
                base_dir = join(config.experiment_execution.paths.experiment_dir_path, "explainability")
                img_dir = join(base_dir, f"fold_{fold_idx}", file_base)
                os.makedirs(img_dir, exist_ok=True)

                save_images(img_dir, "sample", batch[0]['image'][img_idx], cam_fused)

                cam_non_fused_path = join(img_dir, "sample_gradcam_non_fused.png")
                original_np = batch[0]['image'][img_idx].squeeze().cpu().numpy()
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(original_np, cmap='gray')
                ax.imshow(cam_non_fused, cmap='jet', alpha=0.5)
                ax.axis('off')
                ax.set_aspect('equal')
                plt.tight_layout()
                plt.savefig(cam_non_fused_path, dpi=300)
                plt.close()

                # Get model predictions
                fused_logits = fused_model(input_dict)
                non_fused_logits = non_fused_model(input_dict)

                fused_pred = torch.argmax(fused_logits, dim=1).item()
                non_fused_pred = torch.argmax(non_fused_logits, dim=1).item()
                label = batch[1][img_idx].item()

                # Store results for metrics
                all_labels.append(label)
                all_fused_preds.append(fused_pred)
                all_non_fused_preds.append(non_fused_pred)

                # Save metadata for this sample
                metadata = {
                    "file_base": file_base,
                    "fold_idx": fold_idx,
                    "batch_idx": idx,
                    "img_idx": img_idx,
                    "fused_model_prediction": fused_pred,
                    "non_fused_model_prediction": non_fused_pred,
                    "label": label,
                    "timestamp": datetime.now().isoformat(),
                    "input_shape": list(batch[0]['image'][img_idx].shape),
                }
                metadata_path = join(img_dir, "sample_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"Saved outputs for fold {fold_idx}, batch {idx}, image {img_idx}, file {file_base}.")

        # Calculate metrics for this fold
        labels_np = np.array(all_labels)
        fused_preds_np = np.array(all_fused_preds)
        non_fused_preds_np = np.array(all_non_fused_preds)

        def compute_metrics(preds, labels):
            tp = np.sum((preds == 1) & (labels == 1))
            tn = np.sum((preds == 0) & (labels == 0))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            acc = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            auc = roc_auc_score(labels, preds)
            return {
                "auc": auc,
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }

        fused_metrics = compute_metrics(fused_preds_np, labels_np)
        non_fused_metrics = compute_metrics(non_fused_preds_np, labels_np)

        # Save overall metrics
        overall_metadata = {
            "fold_idx": fold_idx,
            "num_batches": idx + 1,
            "num_images": len(all_labels),
            "timestamp": datetime.now().isoformat(),
            "fused_model_ckpt": fused_model_ckpt,
            "non_fused_model_ckpt": non_fused_model_ckpt,
            **{f"fused_model_{k}": v for k, v in fused_metrics.items()},
            **{f"non_fused_model_{k}": v for k, v in non_fused_metrics.items()},
        }

        overall_metadata_path = join(base_dir, f"fold_{fold_idx}", "overall_metadata.json")
        with open(overall_metadata_path, "w") as f:
            json.dump(overall_metadata, f, indent=2)

if __name__ == "__main__":
    run_explainability()