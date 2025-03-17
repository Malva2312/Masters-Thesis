from os.path import abspath, dirname, join
"""
This script sets up the necessary imports and configurations for the main application.
Imports:
    from os.path import abspath, dirname, join:
        - abspath: Returns the absolute path of a given path.
        - dirname: Returns the directory name of a given path.
        - join: Joins one or more path components intelligently.
    import hydra:
        - Hydra is a framework for elegantly configuring complex applications.
    import sys:
        - Provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
    sys.path.append(abspath(join(dirname(__file__), "./data_loading/"))):
        - Adds the data_loading directory to the system path to allow importing modules from it.
    from data_loading.src.modules.data.dataloader.preprocessed_data_loader import LIDCIDRIPreprocessedKFoldDataLoader:
        - Imports the LIDCIDRIPreprocessedKFoldDataLoader class for loading preprocessed data with K-Fold cross-validation.
    from data_loading.src.modules.data.metadata import LIDCIDRIPreprocessedMetaData:
        - Imports the LIDCIDRIPreprocessedMetaData class for handling metadata of the preprocessed data.
    from data_loading.src.modules.utils.paths import PYTHON_PROJECT_DIR_PATH:
        - Imports the PYTHON_PROJECT_DIR_PATH constant which likely holds the path to the Python project directory.
"""
import hydra
import sys

sys.path.append(abspath(join(dirname(__file__), "./data_loading/")))
from data_loading.src.modules.data.dataloader.preprocessed_data_loader import LIDCIDRIPreprocessedKFoldDataLoader
from data_loading.src.modules.data.metadata import LIDCIDRIPreprocessedMetaData
from data_loading.src.modules.utils.paths import PYTHON_PROJECT_DIR_PATH
print(PYTHON_PROJECT_DIR_PATH)

def print_loaded_data_info(k_fold_data_loaders=False, load_mask=False):
    """
    Prints information about the loaded data for lung nodule CT images and their corresponding labels.

    Parameters:
    k_fold_data_loaders (bool): If True, adds an extra indentation to the printed information.
    load_mask (bool): Currently not used in the function.

    Prints:
    - Batch index
    - Data (Lung nodule CT image):
        - Type
        - Shape
        - Min/max values
    - Label (Mean lung nodule malignancy):
        - Type
        - Shape
        - Min/max values
    """
    space = "    " if k_fold_data_loaders else ""
    print(f"{space}    Batch index: {batch_index}")
    print(f"{space}        Data (Lung nodule CT image):")
    print(f"{space}         - Type: {type(data['input_image']).__name__}")
    print(f"{space}         - Shape: {data['input_image'].shape}")
    print(f"{space}         - Min/max values: {data['input_image'].min()}/{data['input_image'].max()}")
    print(f"{space}        Label (Mean lung nodule malignancy)")
    print(f"{space}         - Type: {type(label['lnm']['mean']).__name__}")
    print(f"{space}         - Shape: {label['lnm']['mean'].shape}")
    print(f"{space}         - Min/max values: {label['lnm']['mean'].min()}/{label['lnm']['mean'].max()}")
    
hydra.initialize(config_path='./config', version_base=None)
config = hydra.compose(
    config_name="config", 
    overrides=[ # TODO: add parameterization for changing demo and full data versions
        "data/preprocessed/loader=lidc_idri_preprocessed_data_loader_jn_demo",
        "metadata/preprocessed=lidc_idri_preprocessed_metadata_jn_demo"
    ]
)

metadata = LIDCIDRIPreprocessedMetaData(config=config.metadata.preprocessed)

print("\n-------------------------------------- Demonstrating the K-fold data loader ---------------------------------------\n")
config.data.preprocessed.loader.number_of_k_folds = 5
config.data.preprocessed.loader.test_fraction_of_entire_dataset = None
dataloader = LIDCIDRIPreprocessedKFoldDataLoader(
    config=config.data.preprocessed.loader, 
    lung_nodule_image_metadataframe=metadata.get_lung_nodule_image_metadataframe()
)
data_loaders_by_subset = dataloader.get_data_loaders_by_subset()
for subset_type in ["train", "validation", "test"]:
    print(f"Subset type: {subset_type.title()}")
    for fold_index in range(config.data.preprocessed.loader.number_of_k_folds):
        print(f"    Fold index: {fold_index + 1}")
        for batch_index, (data, label) in enumerate(iter(data_loaders_by_subset[subset_type][fold_index]), 1):
            print_loaded_data_info(k_fold_data_loaders=True)


print("\n-------------------------------------- Demonstrating the Training ---------------------------------------\n")
import torch
import pytorch_lightning as pl
from dev.LungNoduleClassifier import LungNoduleClassifier

# Trainer configuration
trainer = pl.Trainer(
    max_epochs=5,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    log_every_n_steps=10
)

# Initialize model
model = LungNoduleClassifier(lr=1e-4)

# Train the model using the K-fold data loaders
for fold_index in range(config.data.preprocessed.loader.number_of_k_folds):
    print(f"Training on fold {fold_index + 1}")
    train_loader = data_loaders_by_subset["train"][fold_index]
    val_loader = data_loaders_by_subset["validation"][fold_index]
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

def predict(model, image):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0)  # Add batch dimension
        logits = model(image)
        prob = torch.sigmoid(logits).item()
    return "Malignant" if prob > 0.5 else "Benign"

# Example usage with a batch from the test set
test_loader = data_loaders_by_subset["test"][0]
for batch_index, (data, label) in enumerate(test_loader, 1):
    image = data['input_image']
    prediction = predict(model, image)
    print(f"Batch {batch_index} Prediction: {prediction}")
