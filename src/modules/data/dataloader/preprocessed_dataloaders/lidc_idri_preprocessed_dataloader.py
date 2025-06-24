from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader
import numpy
import random
import torch
import torchvision

import os

from src.modules.data.data_augmentation.ct_image_augmenter \
    import CTImageAugmenter
from src.modules.features.feature_extractors import FeatureExtractorManager
import os

def nested_defaultdict():
    return defaultdict(list)

def transpose_if_3d(x):
    return numpy.transpose(x, axes=(1, 2, 0)) if x.ndim == 3 else x

def custom_collate_fn(batch):
    # batch = [(data, label), (data, label), ...]
    file_names, data_dicts, labels = [], [], []
    for item in batch:
        if len(item) == 3:
            file_name, data, label = item
            file_names.append(file_name)
        else:
            data, label = item
        data_dicts.append(data)
        labels.append(label)

    keys = data_dicts[0].keys()
    batched_data = {}
    for key in keys:
        values = [d[key] for d in data_dicts]
        try:
            batched_data[key] = torch.cat(values, dim)
        except RuntimeError:
            # Mantém como lista se não empilhável
            batched_data[key] = values

    if file_names:
        return file_names, batched_data, torch.stack(labels)
    return batched_data, torch.stack(labels)


class LIDCIDRIPreprocessedKFoldDataLoader:
    def __init__(
            self,
            config,
            lung_nodule_image_metadataframe,
            load_data_name=False
    ):
        self.config = config

        self.dataloaders = None
        self.dataloaders_by_subset = None
        self.data_names_by_subset = None
        self.data_splits = None
        self.load_data_name = None
        self.torch_generator = None

        self.dataloaders = defaultdict(list)
        self.dataloaders_by_subset = defaultdict(list)
        self.data_names_by_subset = defaultdict(list)
        self.data_splits = defaultdict(nested_defaultdict)
        self.load_data_name = load_data_name
        self.torch_generator = torch.Generator()

        self.torch_generator.manual_seed(self.config.seed_value)
        self._set_data_splits(lung_nodule_image_metadataframe)
        self._set_dataloaders()

    def get_data_names(self):
        data_names = {subset_type: [
            self.data_splits[subset_type]['file_names'][datafold_id]
            for datafold_id in range(self.config.number_of_k_folds)
        ] for subset_type in ["train", "validation", "test"]}
        return data_names

    def get_dataloaders(self):
        return self.dataloaders

    def _get_torch_dataloader(
            self,
            file_names,
            labels,
            subset_type,
            torch_dataloader_kwargs
    ):
        torch_dataloader = TorchDataLoader(
            dataset=LIDCIDRIPreprocessedDataLoader(
                config=self.config,
                file_names=file_names,
                labels=labels,
                load_data_name=self.load_data_name,
                subset_type=subset_type
            ),
            shuffle=True if subset_type == "train" else False,
            worker_init_fn=self._get_torch_dataloader_worker_init_fn,

            #collate_fn=custom_collate_fn,  # Use custom collate function

            **torch_dataloader_kwargs
        )
        return torch_dataloader

    def _get_torch_dataloader_worker_init_fn(self, worker_id):
        numpy.random.seed(self.config.seed_value + worker_id)
        random.seed(self.config.seed_value + worker_id)

    def _set_dataloaders(self):
        for subset_type in ["train", "validation", "test"]:
            for datafold_id in range(self.config.number_of_k_folds):
                self.dataloaders[subset_type].append(
                    self._get_torch_dataloader(
                        file_names=self.data_splits[subset_type] \
                            ['file_names'][datafold_id],
                        labels=self.data_splits[subset_type] \
                            ['labels'][datafold_id],
                        subset_type=subset_type,
                        torch_dataloader_kwargs=
                            self.config.torch_dataloader_kwargs
                    )
                )

    def _set_data_splits(self, lung_nodule_metadataframe):
        if not self.config.number_of_k_folds:
            train_and_validation_file_name_column, test_file_name_column = \
                train_test_split(
                    lung_nodule_metadataframe,
                    test_size=self.config.test_fraction_of_entire_dataset,
                    random_state=self.config.seed_value,
                    stratify=lung_nodule_metadataframe['label']
                )
            train_file_name_column, validation_file_name_column = \
                train_test_split(
                    train_and_validation_file_name_column,
                    test_size=self.config.validation_fraction_of_train_set,
                    random_state=self.config.seed_value,
                    stratify=lung_nodule_image_metadataframe[
                        lung_nodule_image_metadataframe['file_name'].isin(
                            train_and_validation_file_name_column
                        )
                    ]['Mean Nodule Malignancy'].apply(lambda x: int(x + 0.5))
                )

            self.data_names_by_subset['train'] = \
                train_file_name_column.tolist()
            self.data_names_by_subset['validation'] = \
                validation_file_name_column.tolist()
            self.data_names_by_subset['test'] = \
                test_file_name_column.tolist()

            for subset_type in ["train", "validation", "test"]:
                self.dataloaders_by_subset[subset_type] = \
                    self._get_torch_dataloader(
                        file_names=self.data_names_by_subset[subset_type],
                        label_dataframe=lung_nodule_image_metadataframe,
                        subset_type=subset_type,
                        torch_dataloader_kwargs=
                            self.config.torch_dataloader_kwargs
                    )
        else:
            skf_cross_validator = StratifiedKFold(
                n_splits=self.config.number_of_k_folds,
                shuffle=True,
                random_state=self.config.seed_value
            )
            skf_split_generator = skf_cross_validator.split(
                X=lung_nodule_metadataframe,
                y=lung_nodule_metadataframe['label']
            )

            for datafold_id, (train_and_validation_indexes, test_indexes) \
                    in enumerate(skf_split_generator, 1):
                test_lung_nodule_metadataframe = \
                    lung_nodule_metadataframe.iloc[test_indexes]
                (
                    train_lung_nodule_metadataframe,
                    validation_lung_nodule_metadataframe
                ) = train_test_split(
                    lung_nodule_metadataframe \
                        .iloc[train_and_validation_indexes],
                    test_size=self.config.validation_fraction_of_train_set,
                    random_state=self.config.seed_value,
                    stratify=lung_nodule_metadataframe['label'] \
                        .iloc[train_and_validation_indexes]
                )

                self.data_splits['train']['file_names'].append(
                    train_lung_nodule_metadataframe['file_name'].tolist()
                )
                self.data_splits['train']['labels'].append(
                    train_lung_nodule_metadataframe['label'].tolist()
                )
                self.data_splits['validation']['file_names'].append(
                    validation_lung_nodule_metadataframe['file_name'].tolist()
                )
                self.data_splits['validation']['labels'].append(
                    validation_lung_nodule_metadataframe['label'].tolist()
                )
                self.data_splits['test']['file_names'].append(
                    test_lung_nodule_metadataframe['file_name'].tolist()
                )
                self.data_splits['test']['labels'].append(
                    test_lung_nodule_metadataframe['label'].tolist()
                )


class LIDCIDRIPreprocessedDataLoader(Dataset):
    def __init__(
            self,
            config,
            file_names,
            labels,
            load_data_name,
            subset_type
    ):
        self.config = config
        if config.data_augmentation.apply and subset_type == "train":
            self.file_names = (
                file_names
                + config.data_augmentation.augmented_to_original_data_ratio
                * file_names
            )
        else:
            self.file_names = file_names
        self.labels = labels
        self.load_data_name = load_data_name
        self.subset_type = subset_type

        self.augmented_to_original_data_ratio = \
            config.data_augmentation.augmented_to_original_data_ratio
        self.apply_data_augmentations = config.data_augmentation.apply
        if config.data_augmentation.apply and subset_type == "train":
            self.data_augmenter = CTImageAugmenter(
                parameters=config.data_augmentation.parameters
            )
        self.image_transformer = torchvision.transforms.Compose([
            transpose_if_3d,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.5),
        ])
        self.lnva_names = config.lnva_names
        self.mask_transformer = torchvision.transforms.Compose([
            transpose_if_3d,
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, data_index):
        data = self._get_data(data_index)
        labels = self._get_labels(data_index)
        if not self.load_data_name:
            return data, labels
        else:
            return self.file_names[data_index], data, labels

    def _get_data(self, data_index):
        image = numpy.load(
            "{}/{}.npy".format(
                self.config.image_numpy_arrays_dir_path,
                self.file_names[data_index]
            )
        ).astype(numpy.float32)
        mask = numpy.load(
            "{}/{}.npy".format(
                self.config.mask_numpy_arrays_dir_path,
                self.file_names[data_index]
            )
        ).astype(numpy.float32)
        if self.apply_data_augmentations and data_index >= (
                len(self.file_names)
                / (self.augmented_to_original_data_ratio + 1)
        ) and self.subset_type == "train":
            image, mask = self.data_augmenter(
                image=image,
                mask=mask
            )
        image = self.image_transformer(image)
        mask = self.mask_transformer(mask)
        data = dict(
            image=image,
            mask=mask
        )
        # Path to the handcrafted features directory
        # Get the path to the handcrafted_features directory relative to the script folder
        handcrafted_features_dir = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\data\\features"
        # Ensure handcrafted_features_dir exists before proceeding
        os.makedirs(handcrafted_features_dir, exist_ok=True)
        feature_file_path = "C:\\Users\\janto\\OneDrive\\Ambiente de Trabalho\\data\\features\\{}.pt".format(
            self.file_names[data_index]
        )

        if os.path.exists(feature_file_path):
            # Load handcrafted features if they exist
            handcrafted_features = torch.load(feature_file_path)
            for key, value in handcrafted_features.items():
                data[key] = value
        else:
            # Compute and store handcrafted features
            feature_extractor = FeatureExtractorManager()
            features = feature_extractor(image, mask)
            # Remove batch dimension from each feature and add to data dict
            for key, value in features.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                    data[key] = value.squeeze(0).unsqueeze(2)
                    
                    if value.device != image.device:
                        data[key] = value.to(image.device)
                else:
                    data[key] = value

            # Save the handcrafted features to a file
            torch.save(features, feature_file_path)

        return data

    def _get_labels(self, data_index):
        # nodule_visual_attribute_mean_scores = torch.tensor([
        #     self.label_dataframe.loc[
        #         self.label_dataframe['file_name'] ==
        #             self.file_names[data_index],
        #         f'Mean Nodule {lnva_name.replace("_", " ").title()}'
        #     ].values[0] for lnva_name in self.lnva_names
        # ])
        # statistical_measures_of_nodule_malignancy_scores = dict(
        #     mean=torch.tensor([
        #         self.label_dataframe.loc[
        #             self.label_dataframe['file_name'] ==
        #                 self.file_names[data_index],
        #             'Mean Nodule Malignancy'].values[0]
        #     ]),
        #     std=torch.tensor([
        #         self.label_dataframe.loc[
        #             self.label_dataframe['file_name'] ==
        #                 self.file_names[data_index],
        #             f'Nodule Malignancy StD'].values[0]
        #     ]))
        # labels = dict(
        #     lnva_mean_scores=nodule_visual_attribute_mean_scores,
        #     statistical_measures_of_lnm_scores=
        #         statistical_measures_of_nodule_malignancy_scores
        # )
        labels = torch.tensor([
            float(self.labels[data_index])
        ])
        return labels
