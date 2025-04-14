import pytest

import sys
from os.path import abspath, basename, dirname
src_dir = dirname(abspath(__file__))
while basename(src_dir) != "xai_with_capsule_networks":
    src_dir = dirname(src_dir)
sys.path.append(src_dir)

from src.modules.pytorch_lightning_framework.pytorch_lightning_model import (
        PyTorchLightningEfficientProtoCapsModel
    )
from src.utils.configuration_settings import ConfigurationSettings


@pytest.fixture()
def config():
    config = ConfigurationSettings(read_from_config_used_file=False)
    return config

@pytest.fixture()
def pytorch_lightning_model(config):
    pytorch_lightning_model = PyTorchLightningEfficientProtoCapsModel(
        config=config.pytorch_lightning_model.hyperparameters
    )
    return pytorch_lightning_model

def test_optimisers(pytorch_lightning_model):
    pytorch_lightning_model.configure_optimizers()
    optimiser_parameter_names = (
        pytorch_lightning_model.optimiser_parameter_names
    )
    assert all(
        "prototype" in element for element in
        optimiser_parameter_names['prototype_related']
    )
    assert all(
        "prototype" not in element for element in
        optimiser_parameter_names['non_prototype_related']
    )
