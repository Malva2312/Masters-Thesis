"""Path Management Script for Data Generation and Visualization Project

This script provides utility functions for constructing and managing directory
paths required in a data generation and visualization project.

Constants:
    PREPROCESSED_DATA_DIR_PATH (str): Path to the preprocessed data directory.
    PYTHON_PROJECT_DIR_PATH (str): Path to the Python project directory.
    RAW_DATA_DIR_PATH (str): Path to the raw data directory.
    CURRENT_PROTOCOL_DIR_PATH (str): Path to the current protocol directory.
"""

from os import listdir
from os.path import abspath, dirname, isdir, join
from pathlib import Path
import shutil


def get_python_project_dir_path():
    """Retrieves the path to the Python project directory.

    This function navigates upward from the current file's location until the
    directory name matches the expected project folder name.

    Returns:
        str: The full path to the Python project directory.
    """
    python_project_dir_path = abspath(join(dirname('.'), "../"))
    return python_project_dir_path

def get_preprocessed_data_dir_path(data_storage_source):
    """Retrieves the path to the preprocessed data directory.

    The preprocessed data directory is located within the INESC project
    directory.

    Returns:
        str: The full path to the preprocessed data directory.
    """
    if data_storage_source == "internal":
        preprocessed_data_dir_path = \
            f"{PYTHON_PROJECT_DIR_PATH}/data/preprocessed"
    elif data_storage_source == "INESC's SLURM NAS":
        preprocessed_data_dir_path = \
            f"{INESC_SLURM_NAS_LIDC_IDRI_DATASET_DIR_PATH}/preprocessed_data"
    else:
        raise ValueError(
            f"Invalid data_storage_source value: {data_storage_source}. "
            f"Expected 'internal' or 'INESC's SLURM NAS'."
        )

    return preprocessed_data_dir_path

def get_protocol_specific_preprocessed_data_dir_path(
        data_storage_source, protocol_version
):
    protocol_specific_preprocessed_data_dir_path = "{}/protocol_{}".format(
        get_preprocessed_data_dir_path(data_storage_source),
        protocol_version
    )

    return protocol_specific_preprocessed_data_dir_path

def get_raw_data_dir_path():
    """Retrieves the path to the raw data directory.

    The raw data directory contains the TCIA LIDC-IDRI dataset and is located
    within the INESC project directory.

    Returns:
        str: The full path to the raw data directory.
    """
    raw_data_dir_path = (
        f"{INESC_SLURM_NAS_LIDC_IDRI_DATASET_DIR_PATH}/raw_data"
        f"/TCIA_LIDC-IDRI_20200921/LIDC-IDRI"
    )
    return raw_data_dir_path

if dirname(abspath("")).startswith('/nas-ctm01'):
    INESC_SLURM_NAS_LIDC_IDRI_DATASET_DIR_PATH = \
        "/nas-ctm01/datasets/private/LUCAS/LIDC_IDRI"
else:
    INESC_SLURM_NAS_LIDC_IDRI_DATASET_DIR_PATH = (
        f"{dirname(abspath('')).split('/nas-ctm01')[0]}"
        f"/nas-ctm01/datasets/private/LUCAS/LIDC_IDRI"
    )
PYTHON_PROJECT_DIR_PATH = get_python_project_dir_path()
RAW_DATA_DIR_PATH = get_raw_data_dir_path()

