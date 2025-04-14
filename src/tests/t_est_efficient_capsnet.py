import torch
import pytest

from src.modules.loss_functions.efficient_capsnet_loss_functions import MarginLossFunction, ReconstructionLossFunction, TotalLossFunction


@pytest.fixture
def batch_of_labels():
    return torch.tensor([[1, 3, 0, 1], [3, 3, 2, 1]])


@pytest.fixture
def batch_of_high_level_capsule_magnitudes():
    return torch.tensor([[
        [0.75, 0.5, 0.25, 0.25, 0.5],
        [0.5, 0.5, 0.25, 1.0, 0.05],
        [0.25, 0.05, 0.75, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.75, 0.04]],

        [[0.5, 0.75, 0.25, 0.75, 0.5],
         [0.5, 0.5, 0.25, 1.0, 0.05],
         [0.25, 0.05, 0.75, 0.05, 0.05],
         [0.05, 0.05, 0.05, 0.75, 0.04]]])


@pytest.fixture
def label_weights():
    return torch.tensor([[[1, 2, 1, 5, 1]]])


@pytest.fixture()
def expected_margin_loss():
    return torch.tensor(0.1286)


@pytest.fixture
def batch_of_reconstructed_images():
    return torch.tensor([[
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.8, 1.0, 0.6, 0.0],
        [0.0, 0.8, 1.0, 0.6, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]],

        [[0.0, 0.0, 0.2, 0.0, 0.0],
         [0.0, 0.4, 1.0, 0.6, 0.0],
         [0.0, 0.8, 1.0, 0.6, 0.0],
         [0.0, 0.0, 0.3, 0.0, 0.0]]])


@pytest.fixture
def batch_of_input_images():
    return torch.tensor([[
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]],

        [[0.0, 1.0, 1.0, 1.0, 0.0],
         [0.0, 1.0, 1.0, 1.0, 0.0],
         [0.0, 1.0, 1.0, 1.0, 0.0],
         [0.0, 1.0, 1.0, 1.0, 0.0]]])


@pytest.fixture()
def expected_reconstruction_loss():
    return torch.tensor(0.1562)


@pytest.fixture()
def expected_total_loss():
    return torch.tensor(0.1364)


def test_margin_loss_function(batch_of_labels, batch_of_high_level_capsule_magnitudes, expected_margin_loss, label_weights):
    margin_loss_function = MarginLossFunction(m_positive=0.9, m_negative=0.1, λ=0.5, number_of_classes=5)
    calculated_margin_loss = margin_loss_function(batch_of_labels, batch_of_high_level_capsule_magnitudes, label_weights)
    assert torch.allclose(calculated_margin_loss, expected_margin_loss, atol=1e-4), \
        f"Calculated margin loss does not match expected margin loss"


def test_reconstruction_loss_function(batch_of_reconstructed_images, batch_of_input_images, expected_reconstruction_loss):
    reconstruction_loss_function = ReconstructionLossFunction()
    calculated_reconstruction_loss = reconstruction_loss_function(batch_of_reconstructed_images, batch_of_input_images)
    assert torch.allclose(calculated_reconstruction_loss, expected_reconstruction_loss, atol=1e-4), \
        f"Calculated reconstruction loss does not match expected margin loss"


def test_total_loss_function(
        batch_of_input_images, batch_of_labels, batch_of_reconstructed_images,
        batch_of_high_level_capsule_magnitudes, label_weights, expected_total_loss):
    total_loss_function = TotalLossFunction(m_positive=0.9, m_negative=0.1, λ=0.5, reconstruction_factor=5e-2, number_of_classes=5)
    calculated_total_loss = total_loss_function(
        batch_of_input_images, batch_of_labels, batch_of_reconstructed_images, batch_of_high_level_capsule_magnitudes, label_weights)
    assert torch.allclose(calculated_total_loss, expected_total_loss, atol=1e-4), \
        f"Calculated total loss does not match expected total loss"
