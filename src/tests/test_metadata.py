from os import listdir
import pytest

import sys
from os.path import abspath, basename, dirname
src_dir = dirname(abspath(__file__))
while basename(src_dir) != "xai_with_capsule_networks":
    src_dir = dirname(src_dir)
sys.path.append(src_dir)

from src.modules.data.metadata import MetaData
from src.utils.configuration_settings import ConfigurationSettings
from src.utils.dirs import get_image_numpy_arrays_dir


@pytest.fixture()
def config():
    config = ConfigurationSettings(read_from_config_used_file=False)
    return config

@pytest.fixture()
def metadata(config):
    metadata = MetaData(
        data_preprocessing_protocol_number=
            config.metadata.data_preprocessing_protocol_number
    )
    return metadata

def test_metadata(config, metadata):
    lung_nodule_image_metadataframe = (
        metadata.get_lung_nodule_image_metadataframe()
    )
    image_numpy_arrays_dir = get_image_numpy_arrays_dir(
        data_preprocessing_protocol_number=
            config.metadata.data_preprocessing_protocol_number
    )

    assert lung_nodule_image_metadataframe['nodule_malignancy'].apply(
        lambda x: isinstance(x, list)
    ).all(), (
        f"The type of lung_nodule_image_metadataframe['nodule_malignancy'] "
        f"cells should be list, got "
        f"{type(lung_nodule_image_metadataframe.loc[0, 'nodule_malignancy'])}"
    )
    assert lung_nodule_image_metadataframe['mean_nodule_malignancy'].apply(
        lambda x: 1 <= x <= 5
    ).all(), (
        f"The value of "
        f"lung_nodule_image_metadataframe['mean_nodule_malignancy'] "
        f"cells should be 1 ≤ x ≤ 5"
    )
    assert lung_nodule_image_metadataframe['nodule_malignancy_std'].apply(
        lambda x: 0 <= x <= 2
    ).all(), (
        f"The value of "
         f"lung_nodule_image_metadataframe['nodule_malignancy_std'] "
         f"cells should be 0 ≤ x ≤ 2"
    )
    for lnva_name in config.metadata.lnva.names:
        assert lung_nodule_image_metadataframe[
            f'mean_nodule_{lnva_name}'
        ].apply(lambda x: 1 <= x <= 6).all(), (
            f"The value of "
            f"lung_nodule_image_metadataframe[mean_nodule_{lnva_name}] "
            f"cells should be 1 ≤ x ≤ 6"
        )
    assert set(
        [file_name + ".npy" for file_name in metadata.get_file_names()]
    ).issubset(set(listdir(image_numpy_arrays_dir))), (
        "There are file names in the metadataframe that do not exist"
    )
