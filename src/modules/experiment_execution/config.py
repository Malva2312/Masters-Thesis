from os.path import abspath, dirname
from omegaconf import OmegaConf
from pathlib import Path


class ExperimentExecutionConfig:
    def __init__(self):
        self._experiment_id = None
        self._experiment_version_id = None

    @staticmethod
    def delete_key(config, key):
        OmegaConf.set_struct(config, value=False)
        config.pop(key=key, default=None)
        OmegaConf.set_struct(config, value=True)

    @staticmethod
    def save(config):
        experiment_version_dir_path = \
            config.experiment_execution.paths.experiment_version_dir_path
        OmegaConf.save(
            OmegaConf.to_container(config, resolve=True),
            f=f"{experiment_version_dir_path}/used_config.yaml"
        )

    @staticmethod
    def set_experiment_id(config):
        experiments_dir_path = \
            Path(__file__).resolve().parents[3] / "experiment_results"
        if not any(Path(experiments_dir_path).iterdir()):
            experiment_id = 1
        else:
            experiment_ids = [
                int(folder.name.split("_")[1])
                for folder in Path(experiments_dir_path).iterdir()
                if folder.is_dir()
            ]
            experiment_id = experiment_ids[-1] + 1
        config.experiment_execution.ids.experiment_id = experiment_id

    @staticmethod
    def set_experiment_version_id(config):
        if not config.experiment_execution.ids.experiment_id:
            raise ValueError(
                "It is required to set the experiment ID "
                "before setting the experiment version ID"
            )
        else:
            experiment_dir_path = \
                f"{Path(__file__).resolve().parents[3]}/experiment_results" \
                f"/experiment_{config.experiment_execution.ids.experiment_id}"
            if not Path(experiment_dir_path).exists():
                experiment_version_id = 1
            elif not any(Path(experiment_dir_path).iterdir()):
                experiment_version_id = 1
            else:
                experiment_version_ids = [
                    int(folder.name.split("_")[1])
                    for folder in Path(experiment_dir_path).iterdir()
                    if folder.is_dir()
                ]
                experiment_version_id = experiment_version_ids[-1] + 1
        config.experiment_execution.ids.experiment_version_id = \
            experiment_version_id

    @staticmethod
    def set_hyperparameter_combination_index(
            config,
            hyperparameter_combination_index
    ):
        config.hyperparameter_grid_based_execution.hyperparameter_combination \
            .index = hyperparameter_combination_index

    @staticmethod
    def set_paths(config):
        if config.data.dataset_name == "LIDC-IDRI":
            if dirname(abspath("")).startswith('/nas-ctm01'):
                dataset_dir_path = \
                    "/nas-ctm01/datasets/private/LUCAS/LIDC_IDRI"
            else:
                dataset_dir_path = (
                    f"{dirname(abspath('')).split('/nas-ctm01')[0]}"
                    f"/nas-ctm01/datasets/private/LUCAS/LIDC_IDRI"
                )
        elif config.data.dataset_name == "LUNA25":
            if dirname(abspath("")).startswith('/nas-ctm01'):
                dataset_dir_path = (
                    "/nas-ctm01/datasets/public/medical_datasets"
                    "/lung_ct_datasets/luna25_challenge"
                )
            else:
                dataset_dir_path = (
                    f"{dirname(abspath('')).split('/nas-ctm01')[0]}"
                    f"/nas-ctm01/datasets/public/medical_datasets"
                    f"/lung_ct_datasets/luna25_challenge"
                )
        else:
            raise ValueError(
                "Invalid dataset name: {}. Expected: 'LIDC-IDRI' or 'LUNA25'"
            )

        experiment_id = config.experiment_execution.ids.experiment_id
        experiment_version_id = \
            config.experiment_execution.ids.experiment_version_id

        config.experiment_execution.paths.preprocessed_data_dir_path = \
            f"{dataset_dir_path}/preprocessed_data"
        config.experiment_execution.paths.python_project_dir_path = \
            str(Path(__file__).resolve().parents[3])
        config.experiment_execution.paths.experiment_dir_path = \
            f"{config.experiment_execution.paths.python_project_dir_path}" \
            f"/experiment_results/experiment_{experiment_id}"
        config.experiment_execution.paths.experiment_version_dir_path = \
            f"{config.experiment_execution.paths.experiment_dir_path}" \
            f"/version_{experiment_version_id}"

    @staticmethod
    def set_used_hyperparameter_combination(
            config,
            current_hyperparameter_combination
    ):
        config_string_representations = config \
            .hyperparameter_grid_based_execution.hyperparameter_grid.keys()
        config.hyperparameter_grid_based_execution \
            .hyperparameter_combination.used = OmegaConf.create(
                dict(zip(
                    config_string_representations,
                    current_hyperparameter_combination
                ))
            )

    @staticmethod
    def update_hyperparameter_combination(config):
        for config_string_representation, config_value in (
                config.hyperparameter_grid_based_execution
                        .hyperparameter_combination.used.items()
        ):
            config_keys = config_string_representation.split('.')
            config_copy = config
            for config_key in config_keys[:-1]:
                config_copy = config_copy[config_key]
            config_copy[config_keys[-1]] = config_value

experiment_execution_config = ExperimentExecutionConfig()
