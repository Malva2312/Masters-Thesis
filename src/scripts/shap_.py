from datetime import datetime
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

fused_model_ckpt = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\experiment_48\\version_1\\datafold_5\\models\\mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=32-var=val_auroc=0.924.ckpt"
non_fused_model_ckpt = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\Dissertação\\Masters-Thesis\\data\\experiment_46\\version_1\\datafold_5\\models\\mod=ResNetFusionModel-exp=X-ver=Y-dtf=Z-epoch=39-var=last_epoch.ckpt"
datafold_idx = [4]

class SHAPFeatureWrapper(torch.nn.Module):
    def __init__(self, model, fixed_image_tensor, feature_name):
        super().__init__()
        self.model = model.eval()
        self.fixed_image = fixed_image_tensor
        self.feature_name = feature_name

    def forward(self, features):
        B = features.shape[0]
        images = self.fixed_image.repeat(B, 1, 1, 1)
        input_dict = {'image': images, self.feature_name: features}
        output = self.model(input_dict)
        if output.dim() == 2 and output.shape[1] == 2:
            return output[:, 1]
        return output

def save_shap_plot(save_dir, prefix, shap_vals):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    shap_array = np.array(shap_vals[0])
    if shap_array.ndim == 1:
        ax.barh(range(len(shap_array)), shap_array, color='darkgreen')
    else:
        ax.barh(range(shap_array.shape[1]), shap_array[0], color='darkgreen')
    ax.set_title("SHAP (Handcrafted Features)")
    ax.set_xlabel("SHAP Value")
    ax.set_ylabel("Feature Index")
    plt.tight_layout()
    save_path = join(save_dir, f"{prefix}_shap.png")
    plt.savefig(save_path)
    plt.close()

def save_combined_feature_importance_plot(shap_vals_dict, save_path):
    plt.figure(figsize=(6, 4))
    feature_names = list(shap_vals_dict.keys())
    scores = [np.sum(np.abs(np.array(shap_vals_dict[f][0]))) for f in feature_names]
    plt.barh(feature_names, scores, color='darkred')
    plt.title("Combined SHAP Feature Contributions")
    plt.xlabel("Summed |SHAP Value|")
    plt.tight_layout()
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

    fused_ckpt = torch.load(fused_model_ckpt)
    non_fused_ckpt = torch.load(non_fused_model_ckpt)
    fused_weights = {k.replace("model.", "", 1): v for k, v in fused_ckpt['state_dict'].items()}
    non_fused_weights = {k.replace("model.", "", 1): v for k, v in non_fused_ckpt['state_dict'].items()}

    base_dir = join(config.experiment_execution.paths.experiment_dir_path, "explainability_shap")

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

        for batch_idx, batch in enumerate(test_dataloader):
            for img_idx in range(batch[0]['image'].shape[0]):
                image_tensor = batch[0]['image'][img_idx].unsqueeze(0)
                input_dict_train = {
                    'image': batch[0]['image'],
                    'lbp': batch[0]['lbp'],
                    'shape': batch[0]['shape'],
                    'fof': batch[0]['fof'],
                }

                save_dir_fused = join(base_dir, "fused", f"fold_{fold_idx}", f"batch_{batch_idx}_{img_idx}")
                os.makedirs(save_dir_fused, exist_ok=True)

                shap_vals_dict = {}
                for feature_name in ['lbp', 'fof', 'shape']:
                    feature_tensor = batch[0][feature_name]
                    wrapper = SHAPFeatureWrapper(fused_model, image_tensor, feature_name)
                    background = feature_tensor[:30]
                    test_sample = feature_tensor[img_idx].unsqueeze(0)
                    background_flat = background.view(background.size(0), -1)
                    test_sample_flat = test_sample.view(1, -1)
                    explainer = shap.KernelExplainer(
                        lambda x: wrapper(torch.tensor(x, dtype=torch.float)).detach().numpy(),
                        background_flat.numpy()
                    )
                    shap_vals = explainer.shap_values(test_sample_flat.numpy())
                    shap_vals_dict[feature_name] = shap_vals
                    save_shap_plot(save_dir_fused, feature_name, shap_vals)

                combined_save_path = join(save_dir_fused, "combined_shap_summary.png")
                save_combined_feature_importance_plot(shap_vals_dict, combined_save_path)

                # --- Non-fused model: SHAP for image only (optional) ---
                # save_dir_nf = join(base_dir, "non_fused", f"fold_{fold_idx}", f"batch_{batch_idx}_{img_idx}")
                # os.makedirs(save_dir_nf, exist_ok=True)
                # wrapper_nf = SHAPFeatureWrapper(non_fused_model, image_tensor, "image")
                # background_nf = batch[0]['image'][:30]
                # test_sample_nf = batch[0]['image'][img_idx].unsqueeze(0)
                # background_nf_flat = background_nf.view(background_nf.size(0), -1)
                # test_sample_nf_flat = test_sample_nf.view(1, -1)
                # explainer_nf = shap.KernelExplainer(
                #     lambda x: wrapper_nf(torch.tensor(x, dtype=torch.float)).detach().numpy(),
                #     background_nf_flat.numpy()
                # )
                # shap_vals_nf = explainer_nf.shap_values(test_sample_nf_flat.numpy())
                # save_shap_plot(save_dir_nf, "image", shap_vals_nf)

                print(f"SHAP saved for fold {fold_idx}, batch {batch_idx}, sample {img_idx}.")
                break
            break
        break

if __name__ == "__main__":
    run_explainability()
