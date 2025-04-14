import torch
import pytest

from src.modules.pytorch_lightning_trainer.metrics import PyTorchLightningTrainerMetrics


@pytest.fixture
def random_ground_truth_labels():
    # Generate random ground truth labels (e.g., tensor of shape [batch_size, num_classes])
    return torch.randint(0, 5, (5,))


@pytest.fixture
def random_predicted_scores():
    # Generate random predicted scores (e.g., tensor of shape [batch_size, num_classes])
    return torch.rand(5, 5)


@pytest.fixture
def defined_ground_truth_labels():
    # Define ground truth labels for testing
    return torch.tensor([0, 1, 3, 2])


@pytest.fixture
def defined_predicted_scores():
    # Define predicted scores for testing
    return torch.tensor([
        [0.75, 0.05, 0.05, 0.05, 0.05],
        [0.05, 0.75, 0.05, 0.05, 0.05],
        [0.05, 0.05, 0.75, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.75, 0.04]])


@pytest.fixture
def expected_auroc():
    # Define the multiclass AUROC calculated using both defined ground truth labels and predicted scores
    return torch.tensor([1.0, 1.0, 0.3333, 0.3333, 0.0])


@pytest.fixture
def expected_precision():
    # Define the multiclass Precision calculated using both defined ground truth labels and predicted scores
    return torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])


@pytest.fixture
def expected_specificity():
    # Define the multiclass Specificity calculated using both defined ground truth labels and predicted scores
    return torch.tensor([1.0000, 1.0000, 0.6667, 0.6667, 1.0000])


@pytest.fixture
def expected_recall():
    # Define the multiclass Recall calculated using both defined ground truth labels and predicted scores
    return torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])


def test_auroc(
        random_ground_truth_labels, random_predicted_scores,
        defined_ground_truth_labels, defined_predicted_scores,
        expected_auroc):

    metrics = PyTorchLightningTrainerMetrics()

    # Calculate AUROC from random and defined tensors
    auroc_from_random_tensors = metrics.calculate_auroc(random_predicted_scores, random_ground_truth_labels)
    auroc_from_defined_tensors = metrics.calculate_auroc(defined_predicted_scores, defined_ground_truth_labels)

    # Check if AUROC is within valid range [0, 1]
    assert torch.all(torch.tensor(auroc_from_random_tensors) >= 0) and torch.all(torch.tensor(auroc_from_random_tensors) <= 1)
    assert torch.all(torch.tensor(auroc_from_defined_tensors) >= 0) and torch.all(torch.tensor(auroc_from_defined_tensors) <= 1)

    # Check if AUROC has correct shape
    assert torch.tensor(auroc_from_random_tensors).shape == (random_predicted_scores.shape[-1],)
    assert torch.tensor(auroc_from_defined_tensors).shape == (defined_predicted_scores.shape[-1],)

    # Check if AUROC is being calculated correctly
    assert torch.allclose(expected_auroc, torch.tensor(auroc_from_defined_tensors), atol=1e-4)


def test_precision(
        random_ground_truth_labels, random_predicted_scores,
        defined_ground_truth_labels, defined_predicted_scores,
        expected_precision):

    metrics = PyTorchLightningTrainerMetrics()

    # Calculate Precision from random and defined tensors
    precision_from_random_tensors = metrics.calculate_precision(random_predicted_scores, random_ground_truth_labels)
    precision_from_defined_tensors = metrics.calculate_precision(defined_predicted_scores, defined_ground_truth_labels)

    # Check if Precision is within valid range [0, 1]
    assert torch.all(torch.tensor(precision_from_random_tensors) >= 0) and torch.all(torch.tensor(precision_from_random_tensors) <= 1)
    assert torch.all(torch.tensor(precision_from_defined_tensors) >= 0) and torch.all(torch.tensor(precision_from_defined_tensors) <= 1)

    # Check if Precision has correct shape
    assert torch.tensor(precision_from_random_tensors).shape == (random_predicted_scores.shape[-1],)
    assert torch.tensor(precision_from_defined_tensors).shape == (defined_predicted_scores.shape[-1],)

    # Check if Precision is being calculated correctly
    assert torch.allclose(expected_precision, torch.tensor(precision_from_defined_tensors), atol=1e-4)


def test_specificity(
        random_ground_truth_labels, random_predicted_scores,
        defined_ground_truth_labels, defined_predicted_scores,
        expected_specificity):

    metrics = PyTorchLightningTrainerMetrics()

    # Calculate Specificity from random and defined tensors
    specificity_from_random_tensors = metrics.calculate_specificity(random_predicted_scores, random_ground_truth_labels)
    specificity_from_defined_tensors = metrics.calculate_specificity(defined_predicted_scores, defined_ground_truth_labels)

    # Check if Specificity is within valid range [0, 1]
    assert torch.all(torch.tensor(specificity_from_random_tensors) >= 0) and torch.all(torch.tensor(specificity_from_random_tensors) <= 1)
    assert torch.all(torch.tensor(specificity_from_defined_tensors) >= 0) and torch.all(torch.tensor(specificity_from_defined_tensors) <= 1)

    # Check if Specificity has correct shape
    assert torch.tensor(specificity_from_random_tensors).shape == (random_predicted_scores.shape[-1],)
    assert torch.tensor(specificity_from_defined_tensors).shape == (defined_predicted_scores.shape[-1],)

    # Check if Specificity is being calculated correctly
    assert torch.allclose(expected_specificity, torch.tensor(specificity_from_defined_tensors), atol=1e-4)


def test_recall(
        random_ground_truth_labels, random_predicted_scores,
        defined_ground_truth_labels, defined_predicted_scores,
        expected_recall):

    metrics = PyTorchLightningTrainerMetrics()

    # Calculate Recall from random and defined tensors
    recall_from_random_tensors = metrics.calculate_recall(random_predicted_scores, random_ground_truth_labels)
    recall_from_defined_tensors = metrics.calculate_recall(defined_predicted_scores, defined_ground_truth_labels)

    # Check if Recall is within valid range [0, 1]
    assert torch.all(torch.tensor(recall_from_random_tensors) >= 0) and torch.all(torch.tensor(recall_from_random_tensors) <= 1)
    assert torch.all(torch.tensor(recall_from_defined_tensors) >= 0) and torch.all(torch.tensor(recall_from_defined_tensors) <= 1)

    # Check if Recall has correct shape
    assert torch.tensor(recall_from_random_tensors).shape == (random_predicted_scores.shape[-1],)
    assert torch.tensor(recall_from_defined_tensors).shape == (defined_predicted_scores.shape[-1],)

    # Check if Recall is being calculated correctly
    assert torch.allclose(expected_recall, torch.tensor(recall_from_defined_tensors), atol=1e-4)