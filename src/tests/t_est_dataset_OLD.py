from omegaconf import OmegaConf
from os.path import abspath, dirname
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas
import torch
import pytest

from src.modules.dataset import LIDCIDRIPreprocessedDataLoader
from src.modules.dataset import DataLoader, MetaData

config = OmegaConf.load(f'{dirname(dirname(dirname(abspath(__file__))))}/config/config.yaml')


@pytest.fixture()
def label_column_name():
    label_column_name = "Nodule Malignancy"
    return label_column_name


@pytest.fixture()
def labelling_protocol():
    labelling_protocol = {1.0: 1.0, 1.5: 2.0, 2.0: 2.0, 2.5: 3.0, 3.0: 3.0, 3.5: 4.0, 4.0: 4.0, 4.5: 5.0, 5.0: 5.0}
    return labelling_protocol

# @pytest.fixture
# def data_names():
#     metadata = pandas.read_csv(config.metadata.file_path)
#     data_names = [f"{patient_id}-N{nodule_id}" for patient_id, nodule_id in zip(metadata['Patient ID'].values, metadata['Nodule ID'].values)]
#     return data_names
#
#
# def test_dataloader(data_names):
#     dataloader = DataLoader(
#         data_dir=config.dataloader.data_dir, data_type="train", data_names=data_names, label_type=config.dataloader.label_type,
#         metadata_path=config.dataloader.metadata_path, metadata_label_column_name=config.dataloader.metadata_label_column_name,
#         return_data_name=True, torch_dataloader_kwargs=config.dataloader.torch_dataloader_kwargs)
#
#     for data_name, data, label in dataloader:
#         assert list(data.shape[2:]) == config.tests.input_image_dimension, \
#             f"{data_name} data dimension does not match expected dimension {config.tests.input_image_dimension}"
#         assert torch.all((data >= -1) & (data <= 1)), f"{data_name} data values are not between -1 and 1"
#         assert torch.any(data != 0), f"{data_name} data values are all 0"
#         assert all(element in range(1, config.dataloader.number_of_classes + 1) for element in label)


# def test_lidc_idri_preprocessed_dataloader(data_names):
#     dataloader = LIDCIDRIPreprocessedDataLoader(
#         data_dir=config.dataloader.data_dir, data_names=data_names, label_type=config.dataloader.label_type,
#         metadata_path=config.dataloader.metadata_path, metadata_label_column_name=config.dataloader.metadata_label_column_name,
#         return_data_name=True)
#
#     for data_name, data, label in dataloader:
#         assert list(data.shape[1:]) == config.tests.input_image_dimension, \
#             f"{data_name} data dimension does not match expected dimension {config.tests.input_image_dimension}"
#         assert torch.all((data >= -1) & (data <= 1)), f"{data_name} data values are not between -1 and 1"
#         assert torch.any(data != 0), f"{data_name} data values are all 0"
#         assert label.item() in range(1, config.dataloader.number_of_classes + 1)


def test_metadata(label_column_name, labelling_protocol):
    raw_metadataframe = pandas.read_csv(config.metadata.file_path)
    raw_metadataframe_label_counts = raw_metadataframe[label_column_name].value_counts()

    metadata = MetaData(file_path=config.metadata.file_path, label_column_name=label_column_name, labelling_protocol=labelling_protocol)
    label_dataframe = metadata.get_label_dataframe()
    label_dataframe_label_counts = label_dataframe['Label'].value_counts()

    assert sorted(raw_metadataframe[label_column_name].unique()) == sorted(list(labelling_protocol.keys())), \
        f"Raw metadataframe label values different of those from labelling protocol"
    assert sorted(label_dataframe['Label'].unique()) == sorted(list(set(labelling_protocol.values()))), \
        f"Metadataframe label values different of those from labelling protocol"
    assert raw_metadataframe_label_counts.sum() == label_dataframe_label_counts.sum(), \
        f"Raw metadataframe and metadataframe with different label counts"


def test_stratified_kfold(label_column_name, labelling_protocol):
    metadata = MetaData(file_path=config.metadata.file_path, label_column_name=label_column_name, labelling_protocol=labelling_protocol)
    label_dataframe = metadata.get_label_dataframe()
    skf = StratifiedKFold(n_splits=config.number_of_splits, shuffle=True, random_state=config.seed)
    stratified_kfold_results = dict(
        general=dict(names=label_dataframe['Data name'].tolist(), value_counts=label_dataframe['Label'].value_counts().to_dict()))
    for datafold_id, (train_and_validation_indexes, test_indexes) \
            in enumerate(skf.split(X=label_dataframe['Data name'], y=label_dataframe['Label']), 1):
        train_and_validation_data_names, test_data_names = (
            [label_dataframe['Data name'][train_and_validation_index] for train_and_validation_index in train_and_validation_indexes],
            [label_dataframe['Data name'][test_index] for test_index in test_indexes])
        train_and_validation_labels, test_labels = (
            [label_dataframe['Label'][train_and_validation_index] for train_and_validation_index in train_and_validation_indexes],
            [label_dataframe['Label'][test_index] for test_index in test_indexes])
        train_data_names, validation_data_names, _, _ = train_test_split(
            train_and_validation_data_names, train_and_validation_labels,
            test_size=config.validation_split, stratify=train_and_validation_labels, random_state=config.seed)
        stratified_kfold_results[f'datafold_{datafold_id}'] = dict(
            train_data=dict(
                names=train_data_names,
                value_counts=label_dataframe[label_dataframe['Data name'].isin(train_data_names)]['Label'].value_counts().to_dict()),
            validation_data=dict(
                names=validation_data_names,
                value_counts=label_dataframe[label_dataframe['Data name'].isin(validation_data_names)]['Label'].value_counts().to_dict()),
            test_data=dict(
                names=test_data_names,
                value_counts=label_dataframe[label_dataframe['Data name'].isin(test_data_names)]['Label'].value_counts().to_dict()))
        for data_type in ["train_data", "validation_data", "test_data"]:
            stratified_kfold_results[f'datafold_{datafold_id}'][data_type]['value_ratios'] = {
                label: round(stratified_kfold_results[f'datafold_{datafold_id}'][data_type]['value_counts'][label]
                    / stratified_kfold_results[f'general']['value_counts'][label], 2)
                for label in stratified_kfold_results[f'general']['value_counts'].keys()}

    for datafold_id in range(1, config.number_of_splits + 1):
        for label in stratified_kfold_results[f'general']['value_counts'].keys():
            value_counts_sum = 0
            for data_type in ["train_data", "validation_data", "test_data"]:
                value_counts_sum += stratified_kfold_results[f'datafold_{datafold_id}'][data_type]['value_counts'][label]
            assert stratified_kfold_results[f'general']['value_counts'][label] == value_counts_sum

    def assert_not_present(lists):
        # Check if all elements from the first list are not present in the other lists
        first_list = lists[0]
        other_lists = lists[1:]
        return all((element not in sublist) for sublist in other_lists for element in first_list)

    for datafold_id in range(1, config.number_of_splits + 1):
        assert assert_not_present([
            stratified_kfold_results[f'datafold_{datafold_id}'][data_type]['names']
            for data_type in ["train_data", "validation_data", "test_data"]])

    for data_type in ["test_data"]:
        assert assert_not_present([
            stratified_kfold_results[f'datafold_{datafold_id}'][data_type]['names']
            for datafold_id in range(1, config.number_of_splits + 1)])
