from os.path import abspath, dirname
from pathlib import Path
import numpy
import os
import random
import torch

# _IDS = {}
# _PATHS = {
#     'PYTHON_PROJECT_DIR_PATH': Path(__file__).resolve().parents[3]
# }

def create_experiment_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


# def create_experiment_dirs():
#     Path(_PATHS['EXPERIMENT_DIR_PATH']).mkdir(
#         parents=True, exist_ok=True
#     )
#     Path(_PATHS['EXPERIMENT_VERSION_DIR_PATH']).mkdir(
#         parents=True, exist_ok=True
#     )

def enforce_deterministic_behavior():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def disable_warning_messages():
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = "1"

def get_id(key):
    return _IDS.get(key, None)

def get_path(key):
    return _PATHS.get(key, None)

def set_experiment_id():
    global _IDS

    experiments_dir_path = Path(__file__).resolve().parents[3] / "experiments"
    if not any(Path(experiments_dir_path).iterdir()):
        _IDS['EXPERIMENT_ID'] = 1
    else:
        experiment_ids = [
            int(folder.name.split("_")[1])
            for folder in Path(experiments_dir_path).iterdir()
            if folder.is_dir()
        ]
        _IDS['EXPERIMENT_ID'] = experiment_ids[-1] + 1

def set_experiment_version_id():
    global _IDS

    if 'EXPERIMENT_ID' not in _IDS:
        raise ValueError(
            "It is required to set the experiment ID "
            "before setting the experiment version ID"
        )
    else:
        experiment_dir_path = \
            Path(__file__).resolve().parents[3] \
            / f"experiments/experiment_{_IDS['EXPERIMENT_ID']}"
        if not Path(experiment_dir_path).exists():
            _IDS['EXPERIMENT_VERSION_ID'] = 1
        elif not any(Path(experiment_dir_path).iterdir()):
            _IDS['EXPERIMENT_VERSION_ID'] = 1
        else:
            experiment_version_ids = [
                int(folder.name.split("_")[1])
                for folder in Path(experiment_dir_path).iterdir()
                if folder.is_dir()
            ]
            _IDS['EXPERIMENT_VERSION_ID'] = experiment_version_ids[-1] + 1

def set_paths(dataset_name):
    global _PATHS

    if dataset_name == "LIDC-IDRI":
        if dirname(abspath("")).startswith('/nas-ctm01'):
            _PATHS['DATASET_DIR_PATH'] = \
                "/nas-ctm01/datasets/private/LUCAS/LIDC_IDRI"
        else:
            _PATHS['DATASET_DIR_PATH'] = (
                f"{dirname(abspath('')).split('/nas-ctm01')[0]}"
                f"/nas-ctm01/datasets/private/LUCAS/LIDC_IDRI"
            )
    elif dataset_name == "LUNA25":
        if dirname(abspath("")).startswith('/nas-ctm01'):
            _PATHS['DATASET_DIR_PATH'] = (
                "/nas-ctm01/datasets/public/medical_datasets"
                "/lung_ct_datasets/luna25_challenge"
            )
        else:
            _PATHS['DATASET_DIR_PATH'] = (
                f"{dirname(abspath('')).split('/nas-ctm01')[0]}"
                f"/nas-ctm01/datasets/public/medical_datasets"
                f"/lung_ct_datasets/luna25_challenge"
            )
    else:
        raise ValueError(
            "Invalid dataset name: {}. Expected: 'LIDC-IDRI' or 'LUNA25'"
        )
    _PATHS['PREPROCESSED_DATA_DIR_PATH'] = \
        f"{_PATHS['DATASET_DIR_PATH']}/preprocessed_data"
    _PATHS['EXPERIMENT_DIR_PATH'] = \
        _PATHS['PYTHON_PROJECT_DIR_PATH'] \
        / f"experiments/experiment_{_IDS['EXPERIMENT_ID']}"
    _PATHS['EXPERIMENT_VERSION_DIR_PATH'] = \
        _PATHS['EXPERIMENT_DIR_PATH'] \
        / f"version_{_IDS['EXPERIMENT_VERSION_ID']}"

def set_precision(level="high"):
    """
    Set float32 matrix multiplication precision.
    Available levels: 'high', 'medium', 'low'.
    """
    torch.set_float32_matmul_precision(level)

def set_seed(seed_value):
    """
    Set the seed for reproducibility across various libraries.

    Args:
        seed_value (int): The seed value to ensure reproducibility.
    """
    numpy.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
