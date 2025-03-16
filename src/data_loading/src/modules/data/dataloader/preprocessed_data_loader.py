from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader
import numpy
import random
import torch
import torchvision

from src.modules.data.data_augmentation import CTImageAugmenter
from src.modules.utils.paths import get_preprocessed_data_dir_path


class LIDCIDRIPreprocessedKFoldDataLoader:
    def __init__(
            self,
            config,
            lung_nodule_image_metadataframe,
            load_data_name=False
    ):
        self.config = config
        self.load_data_name = load_data_name

        self.data_loaders_by_subset = defaultdict(list)
        self.data_names_by_subset = defaultdict(list)
        self._torch_generator = None

        self._set_torch_generator()
        self._set_data_names_and_data_loaders(lung_nodule_image_metadataframe)

    def get_data_names_by_subset(self):
        return self.data_names_by_subset

    def get_data_loaders_by_subset(self):
        return self.data_loaders_by_subset

    def _get_torch_data_loader(
            self,
            file_names,
            label_dataframe,
            subset_type,
            torch_dataloader_kwargs
    ):
        if self.config.load_mask:
            mask_numpy_arrays_dir_path = \
                "{}/protocol_{}/mask_numpy_arrays".format(
                    get_preprocessed_data_dir_path(
                        data_storage_source=self.config.data_storage_source
                    ),
                    self.config.data_preprocessing_protocol_number
                )
        else:
            mask_numpy_arrays_dir_path = None
        torch_dataloader = TorchDataLoader(
            dataset=LIDCIDRIPreprocessedDataLoader(
                config=self.config,
                file_names=file_names,
                image_numpy_arrays_dir_path=
                    "{}/protocol_{}/image_numpy_arrays".format(
                        get_preprocessed_data_dir_path(
                            data_storage_source=self.config.data_storage_source
                        ),
                        self.config.data_preprocessing_protocol_number
                    ),
                mask_numpy_arrays_dir_path=mask_numpy_arrays_dir_path,
                label_dataframe=label_dataframe,
                load_data_name=self.load_data_name,
                subset_type=subset_type
            ),
            generator=self._torch_generator,
            shuffle=True if subset_type == "train" else False,
            worker_init_fn=self._get_torch_dataloader_worker_init_fn,
            **torch_dataloader_kwargs
        )
        return torch_dataloader

    def _get_torch_dataloader_worker_init_fn(self, worker_id):
        numpy.random.seed(self.config.seed_value + worker_id)
        random.seed(self.config.seed_value + worker_id)

    def _set_data_names_and_data_loaders(self, lung_nodule_image_metadataframe):
        if not self.config.number_of_k_folds:
            train_and_validation_file_name_column, test_file_name_column = \
                train_test_split(
                    lung_nodule_image_metadataframe['file_name'],
                    test_size=self.config.test_fraction_of_entire_dataset,
                    random_state=self.config.seed_value,
                    stratify=lung_nodule_image_metadataframe[
                        f'Mean Nodule Malignancy'
                    ].apply(lambda x: int(x + 0.5))
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
                self.data_loaders_by_subset[subset_type] = \
                    self._get_torch_data_loader(
                        file_names=self.data_names_by_subset[subset_type],
                        label_dataframe=lung_nodule_image_metadataframe,
                        subset_type=subset_type,
                        torch_dataloader_kwargs=
                            self.config.torch_dataloader_kwargs
                    )
        else:
            stratified_k_fold_cross_validator = StratifiedKFold(
                n_splits=self.config.number_of_k_folds,
                shuffle=True,
                random_state=self.config.seed_value
            )
            for datafold_id, (train_and_validation_indexes, test_indexes) \
                    in enumerate(stratified_k_fold_cross_validator.split(
                        X=lung_nodule_image_metadataframe['file_name'],
                        y=lung_nodule_image_metadataframe[
                            f'Mean Nodule Malignancy'
                        ].apply(lambda x: int(x + 0.5))
                    ),
                    1
            ):
                train_file_name_column, validation_file_name_column = \
                    train_test_split(
                        lung_nodule_image_metadataframe[
                            'file_name'
                        ][train_and_validation_indexes],
                        test_size=self.config.validation_fraction_of_train_set,
                        random_state=self.config.seed_value,
                        stratify=lung_nodule_image_metadataframe[
                            f'Mean Nodule Malignancy'
                        ][train_and_validation_indexes].apply(
                            lambda x: int(x + 0.5)
                        )
                    )

                self.data_names_by_subset['train'].append(
                    train_file_name_column.tolist()
                )
                self.data_names_by_subset['validation'].append(
                    validation_file_name_column.tolist()
                )
                self.data_names_by_subset['test'].append([
                    lung_nodule_image_metadataframe['file_name'][test_index]
                    for test_index in test_indexes
                ])

                for subset_type in ["train", "validation", "test"]:
                    self.data_loaders_by_subset[subset_type].append(
                        self._get_torch_data_loader(
                            file_names=
                                self.data_names_by_subset[subset_type][-1],
                            label_dataframe=lung_nodule_image_metadataframe,
                            subset_type=subset_type,
                            torch_dataloader_kwargs=
                                self.config.torch_dataloader_kwargs
                        )
                    )

    def _set_torch_generator(self):
        self._torch_generator = torch.Generator()
        self._torch_generator.manual_seed(self.config.seed_value)


class LIDCIDRIPreprocessedDataLoader(Dataset):
    def __init__(
            self,
            config,
            file_names,
            image_numpy_arrays_dir_path,
            mask_numpy_arrays_dir_path,
            label_dataframe,
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
        self.label_dataframe = label_dataframe
        self.load_data_name = load_data_name
        self.subset_type = subset_type

        self.augmented_to_original_data_ratio = \
            config.data_augmentation.augmented_to_original_data_ratio
        self.apply_data_augmentations = config.data_augmentation.apply
        if config.data_augmentation.apply and subset_type == "train":
            self.data_augmenter = CTImageAugmenter(
                parameters=config.data_augmentation.parameters
            )
        self.image_numpy_arrays_dir_path = image_numpy_arrays_dir_path
        self.image_transformer = torchvision.transforms.Compose([
            lambda x: numpy.transpose(x, axes=(1, 2, 0))
                if x.ndim == 3 else x,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.5),
        ])
        self.mask_numpy_arrays_dir_path = mask_numpy_arrays_dir_path
        if mask_numpy_arrays_dir_path:
            self.mask_transformer = torchvision.transforms.Compose([
                lambda x: numpy.transpose(x, axes=(1, 2, 0))
                    if x.ndim == 3 else x,
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
                self.image_numpy_arrays_dir_path,
                self.file_names[data_index]
            )
        ).astype(numpy.float32)
        if self.mask_numpy_arrays_dir_path:
            mask = numpy.load(
                "{}/{}.npy".format(
                    self.mask_numpy_arrays_dir_path,
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
            data = dict(
                input_image=self.image_transformer(image),
                input_mask=self.mask_transformer(mask)
            )
        else:
            if self.apply_data_augmentations and data_index >= (
                    len(self.file_names)
                    / (self.augmented_to_original_data_ratio + 1)
            ) and self.subset_type == "train":
                image = self.data_augmenter(image=image)
            data = dict(input_image=self.image_transformer(image))
        return data

    def _get_labels(self, data_index):
        labels = defaultdict(dict)
        if self.config.labels.lnva.mean:
            labels['lnva']['mean'] = torch.tensor([
                self.label_dataframe.loc[
                    self.label_dataframe['file_name'] ==
                        self.file_names[data_index],
                    f'Mean Nodule {lnva_name.replace("_", " ").title()}'
                ].values[0] for lnva_name in self.config.lnva_names
            ])
        if self.config.labels.lnva.std:
            labels['lnva']['std'] = torch.tensor([
                self.label_dataframe.loc[
                    self.label_dataframe['file_name'] ==
                        self.file_names[data_index],
                    f'Nodule {lnva_name.replace("_", " ").title()} StD'
                ].values[0] for lnva_name in self.config.lnva_names
            ])
        if self.config.labels.lnm.mean:
            labels['lnm']['mean'] = torch.tensor([
                self.label_dataframe.loc[
                    self.label_dataframe['file_name'] ==
                    self.file_names[data_index],
                    'Mean Nodule Malignancy'].values[0]
            ])
        if self.config.labels.lnm.std:
            labels['lnm']['std'] = torch.tensor([
                self.label_dataframe.loc[
                    self.label_dataframe['file_name'] ==
                    self.file_names[data_index],
                    f'Nodule Malignancy StD'].values[0]
            ])
        # nodule_visual_attribute_mean_scores = torch.tensor([
        #     self.label_dataframe.loc[
        #         self.label_dataframe['file_name'] ==
        #             self.file_names[data_index],
        #         f'Mean Nodule {lnva_name.replace("_", " ").title()}'
        #     ].values[0] for lnva_name in self.config.lnva_names
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
        return labels
