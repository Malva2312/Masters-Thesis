from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader
import numpy
import random
import torch
import torchvision

from src.modules.data.data_augmentation.ct_image_augmenter \
    import CTImageAugmenter


class LUNA25PreprocessedKFoldDataLoader:
    def __init__(
            self,
            config,
            lung_nodule_metadataframe,
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
        self.data_splits = defaultdict(lambda: defaultdict(list))
        self.load_data_name = load_data_name
        self.torch_generator = torch.Generator()

        self.torch_generator.manual_seed(self.config.seed_value)
        self._set_data_splits(lung_nodule_metadataframe)
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
            dataset=LUNA25PreprocessedDataLoader(
                config=self.config,
                file_names=file_names,
                labels=labels,
                load_data_name=self.load_data_name,
                subset_type=subset_type
            ),
            generator=self.torch_generator,
            shuffle=True if subset_type == "train" else False,
            worker_init_fn=self._get_torch_dataloader_worker_init_fn,
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


class LUNA25PreprocessedDataLoader(Dataset):
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
            lambda x: numpy.transpose(x, axes=(1, 2, 0))
                if x.ndim == 3 else x,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.5),
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, data_index):
        data = self._get_data(data_index)
        label = self._get_label(data_index)
        if not self.load_data_name:
            return data, label
        else:
            return self.file_names[data_index], data, label

    def _get_data(self, data_index):
        image = numpy.load(
            "{}/{}.npy".format(
                self.config.image_numpy_arrays_dir_path,
                self.file_names[data_index]
            )
        ).astype(numpy.float32)
        if self.apply_data_augmentations and data_index >= (
                len(self.file_names)
                / (self.augmented_to_original_data_ratio + 1)
        ) and self.subset_type == "train":
            image = self.data_augmenter(image=image)
        image = self.image_transformer(image)
        data = dict(image=image)

        return data

    def _get_label(self, data_index):
        labels = torch.tensor([
            float(self.labels[data_index])
        ])

        return labels
