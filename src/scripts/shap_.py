from datetime import datetime, UTC
from os.path import abspath, dirname, join
import os
import hydra
import sys
import shap
import torch
import matplotlib.pyplot as plt
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


fused_model_ckpt = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\experiment_48\\version_1\\datafold_5\\models\\mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=57-var=last_epoch.ckpt"
non_fused_model_ckpt = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\experiment_46\\version_1\\datafold_5\\models\\mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=39-var=last_epoch.ckpt"
datafold_idx = [4]


class SHAPFeatureWrapper(torch.nn.Module):
    def __init__(self, model, fixed_image_tensor):
        super().__init__()
        self.model = model.eval()
        self.fixed_image = fixed_image_tensor

    def forward(self, features):
        B = features.shape[0]
        images = self.fixed_image.repeat(B, 1, 1, 1)
        input_dict = {'image': images}
        return self.model(input_dict)

def save_shap_plot(save_dir, prefix, shap_vals):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(shap_vals[0][0])), shap_vals[0][0], color='darkgreen')
    ax.set_title("SHAP (Handcrafted Features)")
    ax.set_xlabel("SHAP Value")
    ax.set_ylabel("Feature Index")
    plt.tight_layout()
    save_path = join(save_dir, f"{prefix}_shap.png")
    plt.savefig(save_path)
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

    base_dir = join(config.experiment_execution.paths.experiment_dir_path, "explainability_shap")

    for fold_idx, test_dataloader in enumerate(kfold_dataloaders['test']):
        if datafold_idx and fold_idx not in datafold_idx:
            continue

        for batch_idx, batch in enumerate(test_dataloader):
            for img_idx in range(batch[0]['image'].shape[0]):

                # Prepare common image
                image_tensor = batch[0]['image'][img_idx].unsqueeze(0).repeat(1, 3, 1, 1)
                handcrafted_tensor = batch[0]['lbp'][img_idx].unsqueeze(0)

                # Prepare training input for model instantiation
                input_dict_train = {
                    'image': batch[0]['image'].repeat(1, 3, 1, 1),
                    'lbp': batch[0]['lbp'].repeat(1, 3, 1, 1),
                    'shape': batch[0]['shape'].repeat(1, 3, 1, 1),
                    'fof': batch[0]['fof'].repeat(1, 3, 1, 1),
                }

                # --- FUSED MODEL ---
                config.model.pytorch_lightning_model.hyperparameters.resnet_config = config.model.pytorch_lightning_model.hyperparameters.fused_resnet_config
                fused_model = ResNet_Fused_Model(config=config.model.pytorch_lightning_model.hyperparameters)
                fused_model(input_dict_train)
                fused_model.load_state_dict(fused_weights, strict=False)
                fused_model.eval()

                fused_wrapper = SHAPFeatureWrapper(fused_model, image_tensor)
                background_fused = batch[0]['lbp'][:30]
                test_sample_fused = handcrafted_tensor

                explainer_fused = shap.KernelExplainer(
                    lambda x: fused_wrapper(torch.tensor(x, dtype=torch.float)).detach().numpy(),
                    background_fused.numpy()
                )
                shap_vals_fused = explainer_fused.shap_values(test_sample_fused.numpy().reshape(1, -1))

                save_dir_fused = join(base_dir, "fused", f"fold_{fold_idx}", f"batch_{batch_idx}_{img_idx}")
                save_shap_plot(save_dir_fused, "sample", shap_vals_fused)

                # --- NON-FUSED MODEL ---
                config.model.pytorch_lightning_model.hyperparameters.resnet_config = config.model.pytorch_lightning_model.hyperparameters.base_resnet_config
                non_fused_model = ResNet_Fused_Model(config=config.model.pytorch_lightning_model.hyperparameters)
                non_fused_model(input_dict_train)
                non_fused_model.load_state_dict(non_fused_weights, strict=False)
                non_fused_model.eval()

                non_fused_wrapper = SHAPFeatureWrapper(non_fused_model, image_tensor)
                background_nf = batch[0]['lbp'][:30]
                test_sample_nf = handcrafted_tensor

                explainer_nf = shap.KernelExplainer(
                    lambda x: non_fused_wrapper(torch.tensor(x, dtype=torch.float)).detach().numpy(),
                    background_nf.numpy()
                )
                shap_vals_nf = explainer_nf.shap_values(test_sample_nf.numpy().reshape(1, -1))

                save_dir_nf = join(base_dir, "non_fused", f"fold_{fold_idx}", f"batch_{batch_idx}_{img_idx}")
                save_shap_plot(save_dir_nf, "sample", shap_vals_nf)

                print(f"SHAP saved for fold {fold_idx}, batch {batch_idx}, sample {img_idx}.")
                break  # Limit to one sample if needed
            break
        break

if __name__ == "__main__":
    run_explainability()
