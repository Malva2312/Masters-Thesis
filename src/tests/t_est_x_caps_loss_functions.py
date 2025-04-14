import torch
import pytest

from src.modules.loss_functions.x_caps_loss_functions import (
    MalignancyScoreLossFunction, ReconstructionRegularisationLossFunction, VisualAttributeLossFunction, TotalLossFunction)
from src.utils.configuration_settings import ConfigurationSettings

configuration_settings = ConfigurationSettings()
config = configuration_settings.all


@pytest.fixture
def batches_of_nodule_malignancy_logit_distributions():
    return [
        torch.tensor(
            data=[[1000., 0., 0., 0., 0.],
                  [0., 0., 1000., 0., 0.]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        torch.tensor(
            data=[[10., 15., 0., 0., 0.],
                  [0., 0., 0., 20., 0.]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))]


@pytest.fixture
def batches_of_actual_nodule_malignancy_mean_scores():
    return [
        torch.tensor(
            data=[[1.],
                  [3.]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        torch.tensor(
            data=[[3.],
                  [2.]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))]


@pytest.fixture
def batches_of_actual_nodule_malignancy_score_stds():
    return torch.tensor(
        data=[[[0.],
               [0.]],
              [[0.2],
               [0.3]]],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


@pytest.fixture()
def batches_of_class_numbers():
    return torch.tensor(
        data=[[1., 2., 3., 4., 5.],
              [1., 2., 3., 4., 5.]],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


@pytest.fixture()
def batches_of_input_images():
    return [
        torch.tensor(
            data=[[[0., 0., 0., 0., 0., 0., 0., 0.],
                   [0.2, 0.9, 0.8, 0.1, 0., 0., 0., 0.],
                   [0., 0.8, 0.7, 0., 0., 0., 0., 0.],
                   [0., 0.1, 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.]],
                  [[0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.9, 0.7, 0.9, 0.],
                   [0., 0., 0., 0., 0.8, 0.6, 0.8, 0.],
                   [0., 0., 0., 0., 0.6, 0.5, 0.9, 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.]]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        torch.tensor(
            data=[[[0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0.2, 0.],
                   [0., 0., 0., 0., 0., 0.9, 0.9, 0.9],
                   [0., 0., 0., 0., 0.2, 0.9, 0.9, 0.9],
                   [0., 0., 0., 0., 0., 0.9, 0.9, 0.9]],
                  [[0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0.9, 0.9, 0., 0., 0.],
                   [0., 0., 0., 0.9, 0.8, 0., 0., 0.],
                   [0., 0., 0., 0.1, 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.]]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))]


@pytest.fixture()
def batches_of_input_masks():
    return [
        torch.tensor(
            data=[[[0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 1., 0., 0., 0., 0., 0.],
                   [0., 1., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.]],
                  [[0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 1., 1., 0.],
                   [0., 0., 0., 0., 1., 1., 1., 0.],
                   [0., 0., 0., 0., 1., 1., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.]]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        torch.tensor(
            data=[[[0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 1., 1.],
                   [0., 0., 0., 0., 0., 1., 1., 1.],
                   [0., 0., 0., 0., 0., 1., 1., 1.]],
                  [[0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 1., 0., 0., 0.],
                   [0., 0., 0., 1., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.]]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))]


@pytest.fixture()
def batches_of_reconstructed_images():
    return [
        torch.tensor(
            data=[[[0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0.6, 0.7, 0., 0., 0., 0., 0.],
                   [0., 0.7, 0.6, 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.]],
                  [[0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.4, 0.1, 0., 0.],
                   [0., 0., 0., 0., 0.8, 0.7, 0.9, 0.],
                   [0., 0., 0., 0., 0.5, 0.9, 0.8, 0.],
                   [0., 0., 0., 0., 0.8, 0.9, 0.8, 0.],
                   [0., 0., 0., 0., 0., 0., 0.1, 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.]]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        torch.tensor(
            data=[[[0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0.1, 0., 0.],
                   [0., 0., 0., 0., 0.2, 0.7, 0.7, 0.6],
                   [0., 0., 0., 0., 0., 0.8, 0.8, 0.9],
                   [0., 0., 0., 0., 0.4, 0.6, 0.8, 0.8]],
                  [[0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0.2, 0.9, 0.8, 0., 0., 0.],
                   [0., 0., 0., 0.8, 0.9, 0.2, 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.]]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))]


@pytest.fixture()
def batches_of_actual_nodule_visual_attribute_mean_scores():
    return [
        torch.tensor(
            data=[[1., 6., 2., 5., 3., 4., 3., 2.],
                  [1., 1., 2., 2., 5., 5., 4., 4.]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        torch.tensor(
            data=[[3., 3., 5., 5., 4., 1., 2., 5.],
                  [5., 5., 4., 5., 3., 5., 2., 2.]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))]


@pytest.fixture()
def batches_of_predicted_nodule_visual_attribute_mean_scores():
    return [
        torch.tensor(
            data=[[2., 3., 2., 1., 5., 4., 4., 6.],
                  [1.5, 3.3, 1., 1.8, 5.5, 3.7, 4.4, 5.6]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        torch.tensor(
            data=[[3.1, 3.2, 4.8, 4.9, 4.3, 1.2, 1.6, 5.1],
                  [4.7, 4.7, 3.8, 5.2, 3.2, 4.8, 1.9, 1.8]],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))]


@pytest.fixture()
def expected_malignancy_score_losses():
    return [torch.tensor(0.), torch.tensor(1.5925)]


@pytest.fixture()
def expected_reconstruction_regularisation_losses():
    return [torch.tensor(0.0203), torch.tensor(0.0469)]


@pytest.fixture()
def expected_visual_attribute_losses():
    return [torch.tensor(11.4), torch.tensor(1.65)]


@pytest.fixture()
def expected_total_losses():
    return [torch.tensor(11.4203), torch.tensor(3.2894)]


def test_malignancy_score_loss_function(
        batches_of_nodule_malignancy_logit_distributions, batches_of_actual_nodule_malignancy_mean_scores,
        batches_of_actual_nodule_malignancy_score_stds, expected_malignancy_score_losses):
    malignancy_score_loss_function = MalignancyScoreLossFunction(unittest=True)
    for (batch_of_predicted_malignancy_scores, batch_of_radiologist_score_means,
         batch_of_radiologist_score_stds, expected_malignancy_score_loss) in zip(
            batches_of_nodule_malignancy_logit_distributions, batches_of_actual_nodule_malignancy_mean_scores,
            batches_of_actual_nodule_malignancy_score_stds, expected_malignancy_score_losses):
        malignancy_score_loss = malignancy_score_loss_function(
            batch_of_predicted_malignancy_scores, batch_of_radiologist_score_means, batch_of_radiologist_score_stds)
        assert torch.allclose(malignancy_score_loss, expected_malignancy_score_loss, atol=1e-4)


def test_visual_attribute_loss_function(
        batches_of_predicted_nodule_visual_attribute_mean_scores,
        batches_of_actual_nodule_visual_attribute_mean_scores,
        expected_visual_attribute_losses):
    visual_attribute_loss_function = VisualAttributeLossFunction(unittest=True)
    for (batch_of_predicted_nodule_visual_attribute_mean_scores,
         batch_of_actual_nodule_visual_attribute_mean_scores, expected_visual_attribute_loss) in zip(
            batches_of_predicted_nodule_visual_attribute_mean_scores,
            batches_of_actual_nodule_visual_attribute_mean_scores, expected_visual_attribute_losses):
        visual_attribute_loss = visual_attribute_loss_function(
            batch_of_predicted_nodule_visual_attribute_mean_scores, batch_of_actual_nodule_visual_attribute_mean_scores)
        assert torch.allclose(visual_attribute_loss, expected_visual_attribute_loss, atol=1e-4)


def test_reconstruction_regularisation_loss_function(
        batches_of_input_images, batches_of_input_masks,
        batches_of_reconstructed_images, expected_reconstruction_regularisation_losses):
    reconstruction_regularisation_loss_function = ReconstructionRegularisationLossFunction(unittest=True)
    for (batch_of_input_images, batch_of_input_masks, batch_of_reconstructed_images,
         expected_reconstruction_regularisation_loss) in zip(
            batches_of_input_images, batches_of_input_masks,
            batches_of_reconstructed_images, expected_reconstruction_regularisation_losses):
        reconstruction_regularisation_loss = reconstruction_regularisation_loss_function(
            batch_of_reconstructed_images, batch_of_input_images, batch_of_input_masks)
        assert torch.allclose(reconstruction_regularisation_loss, expected_reconstruction_regularisation_loss, atol=1e-4)


def test_total_loss_function(
        batches_of_nodule_malignancy_logit_distributions, batches_of_actual_nodule_malignancy_mean_scores,
        batches_of_actual_nodule_malignancy_score_stds, batches_of_predicted_nodule_visual_attribute_mean_scores,
        batches_of_actual_nodule_visual_attribute_mean_scores, batches_of_input_images, batches_of_input_masks,
        batches_of_reconstructed_images, expected_total_losses):
    total_loss_function = TotalLossFunction(unittest=True)
    for (batch_of_nodule_malignancy_logit_distributions, batch_of_actual_nodule_malignancy_mean_scores,
            batch_of_actual_nodule_malignancy_score_stds, batch_of_predicted_nodule_visual_attribute_mean_scores,
            batch_of_actual_nodule_visual_attribute_mean_scores, batch_of_reconstructed_images,
            batch_of_input_images, batch_of_input_masks, expected_total_loss) in zip(
            batches_of_nodule_malignancy_logit_distributions, batches_of_actual_nodule_malignancy_mean_scores,
            batches_of_actual_nodule_malignancy_score_stds, batches_of_predicted_nodule_visual_attribute_mean_scores,
            batches_of_actual_nodule_visual_attribute_mean_scores, batches_of_reconstructed_images,
            batches_of_input_images, batches_of_input_masks, expected_total_losses):
        total_loss = total_loss_function(
            batch_of_nodule_malignancy_logit_distributions,
            batch_of_actual_nodule_malignancy_mean_scores,
            batch_of_actual_nodule_malignancy_score_stds,
            batch_of_predicted_nodule_visual_attribute_mean_scores,
            batch_of_actual_nodule_visual_attribute_mean_scores,
            batch_of_reconstructed_images,
            batch_of_input_images,
            batch_of_input_masks)
        assert torch.allclose(total_loss, expected_total_loss, atol=1e-4)
