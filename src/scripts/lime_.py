from datetime import datetime
from os.path import abspath, dirname, join
import os
import hydra
import sys
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.segmentation import mark_boundaries
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

from lime import lime_image

fused_model_ckpt = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\experiment_48\\version_1\\datafold_5\\models\\mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=32-var=val_auroc=0.924.ckpt"
non_fused_model_ckpt = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\experiment_46\\version_1\\datafold_3\\models\\mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=70-var=last_epoch.ckpt"
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

class LimeExplainer:
    def __init__(self, model, device='cpu', preprocess_fn=None):
        """
        model: a PyTorch model (should output class probabilities)
        device: device to run model on
        preprocess_fn: function to preprocess numpy image for model (should output tensor)
        """
        self.model = model.eval().to(device)
        self.device = device
        self.preprocess_fn = preprocess_fn

    def predict(self, images):
        # images: list of numpy arrays [H,W,3] or [H,W]
        batch = []
        for img in images:
            if self.preprocess_fn is not None:
                tensor = self.preprocess_fn(img)
            else:
                tensor = torch.tensor(img).float().unsqueeze(0)
            batch.append(tensor)
        batch_tensor = torch.cat(batch, dim=0).to(self.device)
        with torch.no_grad():
            outputs = self.model({'image': batch_tensor})
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.cpu().numpy()
            else:
                outputs = outputs['logits'].cpu().numpy()
        # Return probabilities
        if outputs.shape[-1] == 1:
            # Binary (logits)
            probs = 1 / (1 + np.exp(-outputs))
            return np.hstack([1 - probs, probs])
        else:
            # Multiclass
            return torch.nn.functional.softmax(torch.from_numpy(outputs), dim=1).numpy()

    def explain_image(self, np_img, label, num_samples=1000):
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np_img.astype(np.double),
            classifier_fn=self.predict,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )
        temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=8, hide_rest=False)
        return temp, mask, explanation

    def save_lime_visualization(self, np_img, mask, out_path):
        plt.figure(figsize=(5,5))
        plt.imshow(mark_boundaries(np_img.squeeze(), mask))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

def preprocess_fn(img):
    # img: numpy array, shape (H, W, 3) or (H, W)
    # If 2D, convert to 3D (repeat channels)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = torch.tensor(img).float().unsqueeze(0)
    return img

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

    fused_ckpt = torch.load(fused_model_ckpt, map_location='cpu')
    non_fused_ckpt = torch.load(non_fused_model_ckpt, map_location='cpu')
    fused_weights = {k.replace("model.", "", 1): v for k, v in fused_ckpt['state_dict'].items()}
    non_fused_weights = {k.replace("model.", "", 1): v for k, v in non_fused_ckpt['state_dict'].items()}

    device = 'cpu'#= 'cuda' if torch.cuda.is_available() else 'cpu'

    for fold_idx, test_dataloader in enumerate(kfold_dataloaders['test']):
        if datafold_idx and fold_idx not in datafold_idx:
            continue

        all_labels = []
        all_fused_preds = []
        all_non_fused_preds = []

        for idx, batch in enumerate(test_dataloader):
            batch_size = batch[0]['image'].shape[0]
            labels = batch[1]

            input_dict_train = {
                'image': batch[0]['image'].repeat(1, 1, 1, 1).to(device),
                'lbp': batch[0]['lbp'].repeat(1, 1, 1, 1).to(device),
                'shape': batch[0]['shape'].repeat(1, 1, 1, 1).to(device),
                'fof': batch[0]['fof'].repeat(1, 1, 1, 1).to(device),
            }

            # Fused model
            config.model.pytorch_lightning_model.hyperparameters.resnet_config = config.model.pytorch_lightning_model.hyperparameters.fused_resnet_config
            fused_model = ResNet_Fused_Model(config=config.model.pytorch_lightning_model.hyperparameters)
            fused_model = fused_model.to(device)
            fused_model(input_dict_train)  # Call after moving to device
            fused_model.load_state_dict(fused_weights, strict=True)
            fused_model.eval()

            # Non-fused model
            config.model.pytorch_lightning_model.hyperparameters.resnet_config = config.model.pytorch_lightning_model.hyperparameters.base_resnet_config
            non_fused_model = ResNet_Fused_Model(config=config.model.pytorch_lightning_model.hyperparameters)
            non_fused_model = non_fused_model.to(device)
            non_fused_model.load_state_dict(non_fused_weights, strict=True)
            non_fused_model.eval()

            # LIME explainer for fused model (initialized here to ensure fresh model)
            lime_explainer = LimeExplainer(fused_model, device=device, preprocess_fn=preprocess_fn)
            # LIME explainer for non-fused model
            lime_explainer_non_fused = LimeExplainer(non_fused_model, device=device, preprocess_fn=preprocess_fn)

            for img_idx in range(batch_size):
                input_dict = {
                    'image': batch[0]['image'][img_idx].unsqueeze(0).repeat(1, 1, 1, 1),
                    'lbp': batch[0]['lbp'][img_idx].unsqueeze(0).repeat(1, 1, 1, 1),
                    'shape': batch[0]['shape'][img_idx].unsqueeze(0).repeat(1, 1, 1, 1),
                    'fof': batch[0]['fof'][img_idx].unsqueeze(0).repeat(1, 1, 1, 1),
                }

                file_base = batch[2][img_idx]

                cam_fused = GradCAM(fused_model, fused_model.resnet_model.model.layer2).generate_cam(input_dict)
                cam_non_fused = GradCAM(non_fused_model, non_fused_model.resnet_model.model.layer2).generate_cam(input_dict)

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

                # LIME explainability
                np_img = batch[0]['image'][img_idx].squeeze().cpu().numpy()
                if np_img.ndim == 2:
                    np_img_3c = np.stack([np_img]*3, axis=-1)
                else:
                    np_img_3c = np_img
                label = batch[1][img_idx].item()
                try:
                    temp, mask, explanation = lime_explainer.explain_image(np_img_3c, label=1)
                    lime_path = join(img_dir, "sample_lime.png")
                    lime_explainer.save_lime_visualization(np_img_3c, mask, lime_path)

                    temp_non_fused, mask_non_fused, explanation_non_fused = lime_explainer_non_fused.explain_image(np_img_3c, label=1)
                    lime_non_fused_path = join(img_dir, "sample_lime_non_fused.png")
                    lime_explainer_non_fused.save_lime_visualization(np_img_3c, mask_non_fused, lime_non_fused_path)
                except Exception as e:
                    print(f"LIME explanation failed for {file_base}: {e}")

                with torch.no_grad():
                    fused_logits = fused_model(input_dict)
                    non_fused_logits = non_fused_model(input_dict)
                    fused_pred = torch.argmax(fused_logits, dim=1, keepdim=True)
                    non_fused_pred = torch.argmax(non_fused_logits, dim=1, keepdim=True)

                all_labels.append(label)
                all_fused_preds.append(fused_pred)
                all_non_fused_preds.append(non_fused_pred)

                metadata = {
                    "file_base": file_base,
                    "fold_idx": fold_idx,
                    "batch_idx": idx,
                    "img_idx": img_idx,
                    "fused_model_prediction": int(fused_pred),
                    "non_fused_model_prediction": int(non_fused_pred),
                    "label": label,
                    "timestamp": datetime.now().isoformat(),
                    "input_shape": list(batch[0]['image'][img_idx].shape),
                }
                metadata_path = join(img_dir, "sample_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"Saved outputs for fold {fold_idx}, batch {idx}, image {img_idx}, file {file_base}.")

        # Convert lists to numpy arrays for logical operations
        labels_np = np.array(all_labels)
        fused_preds_np = np.array(all_fused_preds)
        non_fused_preds_np = np.array(all_non_fused_preds)

        # True/False Positives/Negatives for fused model
        tp_fused = np.sum((fused_preds_np == 1) & (labels_np == 1))
        tn_fused = np.sum((fused_preds_np == 0) & (labels_np == 0))
        fp_fused = np.sum((fused_preds_np == 1) & (labels_np == 0))
        fn_fused = np.sum((fused_preds_np == 0) & (labels_np == 1))

        # True/False Positives/Negatives for non-fused model
        tp_non_fused = np.sum((non_fused_preds_np == 1) & (labels_np == 1))
        tn_non_fused = np.sum((non_fused_preds_np == 0) & (labels_np == 0))
        fp_non_fused = np.sum((non_fused_preds_np == 1) & (labels_np == 0))
        fn_non_fused = np.sum((non_fused_preds_np == 0) & (labels_np == 1))

        # Metrics for fused model
        fused_acc = (tp_fused + tn_fused) / (tp_fused + tn_fused + fp_fused + fn_fused)
        fused_precision = tp_fused / (tp_fused + fp_fused) if (tp_fused + fp_fused) > 0 else 0
        fused_recall = tp_fused / (tp_fused + fn_fused) if (tp_fused + fn_fused) > 0 else 0
        fused_f1 = 2 * fused_precision * fused_recall / (fused_precision + fused_recall) if (fused_precision + fused_recall) > 0 else 0
        fused_auc = roc_auc_score(labels_np.flatten(), fused_preds_np.flatten())

        # Metrics for non-fused model
        non_fused_acc = (tp_non_fused + tn_non_fused) / (tp_non_fused + tn_non_fused + fp_non_fused + fn_non_fused)
        non_fused_precision = tp_non_fused / (tp_non_fused + fp_non_fused) if (tp_non_fused + fp_non_fused) > 0 else 0
        non_fused_recall = tp_non_fused / (tp_non_fused + fn_non_fused) if (tp_non_fused + fn_non_fused) > 0 else 0
        non_fused_f1 = 2 * non_fused_precision * non_fused_recall / (non_fused_precision + non_fused_recall) if (non_fused_precision + non_fused_recall) > 0 else 0
        non_fused_auc = roc_auc_score(labels_np.flatten(), non_fused_preds_np.flatten())

        overall_metadata = {
            "fold_idx": fold_idx,
            "num_batches": idx + 1,
            "num_images": len(all_labels),
            "timestamp": datetime.now().isoformat(),
            "fused_model_ckpt": fused_model_ckpt,
            "non_fused_model_ckpt": non_fused_model_ckpt,
            "fused_model_auc": fused_auc,
            "fused_model_acc": fused_acc,
            "fused_model_precision": fused_precision,
            "fused_model_recall": fused_recall,
            "fused_model_f1": fused_f1,
            "fused_model_tp": int(tp_fused),
            "fused_model_tn": int(tn_fused),
            "fused_model_fp": int(fp_fused),
            "fused_model_fn": int(fn_fused),
            "non_fused_model_auc": non_fused_auc,
            "non_fused_model_acc": non_fused_acc,
            "non_fused_model_precision": non_fused_precision,
            "non_fused_model_recall": non_fused_recall,
            "non_fused_model_f1": non_fused_f1,
            "non_fused_model_tp": int(tp_non_fused),
            "non_fused_model_tn": int(tn_non_fused),
            "non_fused_model_fp": int(fp_non_fused),
            "non_fused_model_fn": int(fn_non_fused),
        }

        overall_metadata_path = join(base_dir, f"fold_{fold_idx}", "overall_metadata.json")
        with open(overall_metadata_path, "w") as f:
            json.dump(overall_metadata, f, indent=2)

if __name__ == "__main__":
    run_explainability()