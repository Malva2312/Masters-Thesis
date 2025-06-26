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
from sklearn.metrics import roc_auc_score
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

from captum.attr import IntegratedGradients

fused_model_ckpt = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\experiment_48\\version_1\\datafold_5\\models\\mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=32-var=val_auroc=0.924.ckpt"
non_fused_model_ckpt = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\experiment_46\\version_1\\datafold_3\\models\\mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=70-var=last_epoch.ckpt"
datafold_idx = [4]

def save_images(base_path, name_prefix, image_tensor, ig_array):
    os.makedirs(base_path, exist_ok=True)
    original_path = join(base_path, f"{name_prefix}_original.png")
    zoom_path = join(base_path, f"{name_prefix}_zoom.png")
    ig_overlay_path = join(base_path, f"{name_prefix}_integrated_gradients.png")
    original_np = image_tensor.squeeze().cpu().numpy()
    plt.imsave(original_path, original_np, cmap='gray')
    zoomed_resized = resize(original_np, (512, 512), order=0, mode='reflect', anti_aliasing=False, preserve_range=True)
    plt.imsave(zoom_path, zoomed_resized.astype(original_np.dtype), cmap='gray')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(original_np, cmap='gray')
    ax.imshow(ig_array, cmap='hot', alpha=0.5)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(ig_overlay_path, dpi=300)
    plt.close()

def get_model_pred_fn(model, device):
    def pred_fn(x):
        model.eval()
        with torch.no_grad():
            input_dict = {'image': x.to(device)}
            logits = model(input_dict)
            probs = torch.softmax(logits, dim=1)
        return probs
    return pred_fn

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                'image': batch[0]['image'].repeat(1, 1, 1, 1),
                'lbp': batch[0]['lbp'].repeat(1, 1, 1, 1),
                'shape': batch[0]['shape'].repeat(1, 1, 1, 1),
                'fof': batch[0]['fof'].repeat(1, 1, 1, 1),
            }

            # Fused model
            config.model.pytorch_lightning_model.hyperparameters.resnet_config = config.model.pytorch_lightning_model.hyperparameters.fused_resnet_config
            fused_model = ResNet_Fused_Model(config=config.model.pytorch_lightning_model.hyperparameters)
            fused_model(input_dict_train)
            fused_model.load_state_dict(fused_weights, strict=True)
            fused_model.to(device)
            fused_model.eval()
            # Non-fused model
            config.model.pytorch_lightning_model.hyperparameters.resnet_config = config.model.pytorch_lightning_model.hyperparameters.base_resnet_config
            non_fused_model = ResNet_Fused_Model(config=config.model.pytorch_lightning_model.hyperparameters)
            non_fused_model.load_state_dict(non_fused_weights, strict=True)
            non_fused_model.to(device)
            non_fused_model.eval()

            # Integrated Gradients for fused model
            ig_fused = IntegratedGradients(lambda x: fused_model({'image': x}))
            ig_non_fused = IntegratedGradients(lambda x: non_fused_model({'image': x}))

            for img_idx in range(batch_size):
                input_dict = {
                    'image': batch[0]['image'][img_idx].unsqueeze(0).to(device),
                    'lbp': batch[0]['lbp'][img_idx].unsqueeze(0).to(device),
                    'shape': batch[0]['shape'][img_idx].unsqueeze(0).to(device),
                    'fof': batch[0]['fof'][img_idx].unsqueeze(0).to(device),
                }

                file_base = batch[2][img_idx]

                image_tensor = batch[0]['image'][img_idx].unsqueeze(0).to(device)
                label = int(batch[1][img_idx].item())  # Ensure integer
                
                # Reference baseline: all zeros (black image)
                baseline = torch.zeros_like(image_tensor).to(device)

                # Fused model Integrated Gradients
                attributions_fused = ig_fused.attribute(
                    image_tensor, baselines=baseline, target=label, n_steps=50
                )
                ig_fused_np = attributions_fused.squeeze().abs().cpu().numpy()
                ig_fused_np = (ig_fused_np - ig_fused_np.min()) / (ig_fused_np.max() - ig_fused_np.min() + 1e-8)

                base_dir = join(config.experiment_execution.paths.experiment_dir_path, "explainability")
                img_dir = join(base_dir, f"fold_{fold_idx}", file_base)
                os.makedirs(img_dir, exist_ok=True)
                save_images(img_dir, "sample", batch[0]['image'][img_idx], ig_fused_np)

                # Non-fused model Integrated Gradients
                attributions_non_fused = ig_non_fused.attribute(
                    image_tensor, baselines=baseline, target=label, n_steps=50
                )
                ig_non_fused_np = attributions_non_fused.squeeze().abs().cpu().numpy()
                ig_non_fused_np = (ig_non_fused_np - ig_non_fused_np.min()) / (ig_non_fused_np.max() - ig_non_fused_np.min() + 1e-8)
                ig_non_fused_path = join(img_dir, "sample_integrated_gradients_non_fused.png")
                original_np = batch[0]['image'][img_idx].squeeze().cpu().numpy()
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(original_np, cmap='gray')
                ax.imshow(ig_non_fused_np, cmap='hot', alpha=0.5)
                ax.axis('off')
                ax.set_aspect('equal')
                plt.tight_layout()
                plt.savefig(ig_non_fused_path, dpi=300)
                plt.close()

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

        # Metrics for fused model
        fused_auc = roc_auc_score(labels_np.flatten(), fused_preds_np.flatten())
        fused_acc = (fused_preds_np == labels_np).mean()

        # Metrics for non-fused model
        non_fused_auc = roc_auc_score(labels_np.flatten(), non_fused_preds_np.flatten())
        non_fused_acc = (non_fused_preds_np == labels_np).mean()

        overall_metadata = {
            "fold_idx": fold_idx,
            "num_batches": idx + 1,
            "num_images": len(all_labels),
            "timestamp": datetime.now().isoformat(),
            "fused_model_ckpt": fused_model_ckpt,
            "non_fused_model_ckpt": non_fused_model_ckpt,
            "fused_model_auc": fused_auc,
            "fused_model_acc": fused_acc,
            "non_fused_model_auc": non_fused_auc,
            "non_fused_model_acc": non_fused_acc,
        }

        overall_metadata_path = join(base_dir, f"fold_{fold_idx}", "overall_metadata.json")
        with open(overall_metadata_path, "w") as f:
            json.dump(overall_metadata, f, indent=2)

if __name__ == "__main__":
    run_explainability()