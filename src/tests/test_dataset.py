import math
import pytest
import torch

import sys
from os.path import abspath, basename, dirname
src_dir = dirname(abspath(__file__))
while basename(src_dir) != "xai_with_capsule_networks":
    src_dir = dirname(src_dir)
sys.path.append(src_dir)

from src.modules.data.dataset import DataLoader
from src.modules.data.metadata import MetaData
from src.utils.configuration_settings import ConfigurationSettings


@pytest.fixture()
def config():
    config = ConfigurationSettings(read_from_config_used_file=False)
    return config

@pytest.fixture()
def dataloader(config):
    dataloader = DataLoader(
        config=config.dataloader,
        return_data_name=True
    )
    return dataloader

@pytest.fixture()
def lung_nodule_image_metadataframe(config):
    metadata = MetaData(
        data_preprocessing_protocol_number=
            config.dataloader.data_preprocessing_protocol_number
    )
    lung_nodule_image_metadataframe = (
        metadata.get_lung_nodule_image_metadataframe()
    )
    lung_nodule_image_metadataframe[f'mean_nodule_malignancy'] = \
        lung_nodule_image_metadataframe[f'mean_nodule_malignancy'].apply(
            lambda x: int(x + 0.5))
    return lung_nodule_image_metadataframe


def test_dataloader(config, dataloader, lung_nodule_image_metadataframe):

    k_fold_dataloaders = dataloader.get_k_fold_dataloaders()
    k_fold_data_names = dataloader.get_k_fold_data_names()
    for k in range(5):
        expected_data_split_fraction = dict(
            train=0.72,
            validation=0.08,
            test=0.2
        )
        difference_tolerance = 0.05

        current_fold_data_names = dict(all=[])
        for data_type in ['train', 'validation', 'test']:
            current_fold_data_names[data_type] = (
                k_fold_data_names[data_type][k]
            )
            current_fold_data_names['all'] += k_fold_data_names[data_type][k]

        for data_type in ['train', 'validation', 'test']:
            actual_data_split_fraction = (
                    len(current_fold_data_names[data_type]) / len(
                current_fold_data_names['all'])
            )

            assert math.isclose(
                a=actual_data_split_fraction,
                b=expected_data_split_fraction[data_type],
                rel_tol=difference_tolerance
            ), "{} split percentage is incorrect. Expected {}, got {}".format(
                data_type.title(),
                expected_data_split_fraction[data_type],
                actual_data_split_fraction
            )

        assert set(k_fold_data_names['test'][k]).isdisjoint(
            set(k_fold_data_names['train'][k])) \
               and set(k_fold_data_names['test'][k]).isdisjoint(
            set(k_fold_data_names['validation'][k]))

        for remaining_k in range(k + 1, 5):
            if not set(k_fold_data_names['test'][k]).isdisjoint(
                    set(k_fold_data_names['test'][remaining_k])):
                conflicting_data_names = (
                    set(k_fold_data_names['test'][k]).intersection(
                        set(k_fold_data_names['test'][remaining_k]))
                )
                raise AssertionError(
                    f"Conflict found: Test data names {conflicting_data_names}"
                    f" are present in both folds "
                    f"{k + 1} and {remaining_k + 1}."
                    f"\nFold {k + 1} test data names: "
                    f"{k_fold_data_names['test'][k]}, "
                    f"\nFold {remaining_k + 1} test data names: "
                    f"{k_fold_data_names['test'][remaining_k]}"
                )

        normalised_mean_lnm_score_counts = {
            data_type: lung_nodule_image_metadataframe[
                lung_nodule_image_metadataframe['file_name'].isin(
                    k_fold_data_names[data_type][k])
            ][
                f'mean_nodule_malignancy'].value_counts().sort_index() /
            lung_nodule_image_metadataframe[
                lung_nodule_image_metadataframe['file_name'].isin(
                    k_fold_data_names[data_type][k])
            ][f'mean_nodule_malignancy'].value_counts().max()
            for data_type in ['train', 'validation', 'test']
        }
        normalised_mean_lnm_score_counts['all'] = (
            lung_nodule_image_metadataframe[
                f'mean_nodule_malignancy'
            ].value_counts().sort_index() / lung_nodule_image_metadataframe[
                f'mean_nodule_malignancy'
            ].value_counts().max()
        )

        for (
                train_lnm_score_count,
                validation_lnm_score_count,
                test_lnm_score_count,
                all_lnm_score_count
        ) in zip(
                normalised_mean_lnm_score_counts['train'],
                normalised_mean_lnm_score_counts['validation'],
                normalised_mean_lnm_score_counts['test'],
                normalised_mean_lnm_score_counts['all']
        ):
            assert math.isclose(
                all_lnm_score_count, train_lnm_score_count,
                rel_tol=difference_tolerance
            )
            assert math.isclose(
                all_lnm_score_count, validation_lnm_score_count,
                rel_tol=difference_tolerance
            )
            assert math.isclose(
                all_lnm_score_count, test_lnm_score_count,
                rel_tol=difference_tolerance
            )

    for data_type in ['train', 'validation', 'test']:
        for dataloader in k_fold_dataloaders[data_type]:
            for data_name, data, label in dataloader:
                assert (
                    list(data['input_image'].shape[1:])
                    == config.dataloader.input_data_size
                ), (
                    f"{data_name} image dimension does not "
                    f"match expected dimension "
                    f"{config.dataloader.input_data_size}"
                )
                assert list(data['input_mask'].shape[1:]) == config.dataloader.input_data_size, \
                    f"{data_name} mask dimension does not match expected dimension {config.dataloader.input_data_size}"
                assert torch.all((data['input_image'] >= -1) & (data['input_image'] <= 1)), \
                    f"{data_name} image values are not between -1 and 1"
                assert torch.any(data['input_image'] != 0), \
                    f"{data_name} image values are all 0"
                assert torch.all((data['input_mask'] == 0) | (data['input_mask'] == 1)), \
                    f"{data_name} image values are not between -1 and 1"
                assert torch.all(
                    (label['lnva_mean_scores'] >= 1) & (label['lnva_mean_scores'] <= 6)), \
                    f"{data_name} nodule visual attribute mean scores are not between 1 and 6"
                assert torch.all(
                    (label['statistical_measures_of_lnm_scores']['mean'] >= 1)
                    & (label['statistical_measures_of_lnm_scores']['mean'] <= 5)), \
                    f"{data_name} nodule malignancy mean scores are not between 1 and 5"
