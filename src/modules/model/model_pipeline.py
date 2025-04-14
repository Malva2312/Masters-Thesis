from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import json
import os

from src.modules.model.pytorch_lightning_model import PyTorchLightningModel


class ModelPipeline:
    def __init__(
            self,
            config,
            datafold_id,
            dataloaders,
            data_file_names,
            experiment_execution_ids,
            experiment_execution_paths
    ):
        self.config = config

        self.data_file_names = data_file_names
        self.datafold_id = datafold_id
        self.dataloaders = dataloaders
        self.experiment_id = experiment_execution_ids.experiment_id
        self.experiment_version_id = \
            experiment_execution_ids.experiment_version_id
        self.experiment_dir_path = \
            experiment_execution_paths.experiment_dir_path
        self.experiment_version_dir_path = \
            experiment_execution_paths.experiment_version_dir_path
        self.pytorch_lightning_model = PyTorchLightningModel(
            config=self.config.pytorch_lightning_model,
            experiment_execution_paths=experiment_execution_paths
        )
        self.pytorch_lightning_trainer = Trainer(
            callbacks=self._get_model_trainer_callbacks(),
            logger=self._get_model_trainer_loggers(),
            **self.config.pytorch_lightning_trainer_kwargs
        )

    def delete_model_checkpoints(self):
        for trainer_callback in self.pytorch_lightning_trainer.callbacks:
            if isinstance(trainer_callback, ModelCheckpoint):
                os.remove(path=trainer_callback.best_model_path)
        os.rmdir(
            f"{self.experiment_version_dir_path}"
            f"/datafold_{self.datafold_id}/models"
        )

    def finalize(self):
        if self.config.finalization.delete_model_checkpoints:
            self.delete_model_checkpoints()
        if self.config.finalization.save_used_data_file_names:
            self.save_used_data_file_names()

    def save_used_data_file_names(self):
        with open(
                f"{self.experiment_version_dir_path}"
                f"/datafold_{self.datafold_id}/used_data_file_names.json",
                'w'
        ) as file:
            json.dump(obj=self.data_file_names, fp=file)

    def train_model(self):
        self.pytorch_lightning_trainer.fit(
            model=self.pytorch_lightning_model,
            train_dataloaders=self.dataloaders['train'],
            val_dataloaders=self.dataloaders['validation']
        )

    def test_model(self):
        for trainer_callback in self.pytorch_lightning_trainer.callbacks:
            if isinstance(trainer_callback, ModelCheckpoint):
                self.pytorch_lightning_trainer.test(
                    ckpt_path=trainer_callback.best_model_path,
                    dataloaders=self.dataloaders['test'],
                    verbose=False
                )

    def _get_model_trainer_callbacks(self):
        trainer_callbacks = []
        if self.config.pytorch_lightning_trainer_kwargs.enable_checkpointing:
            for model_checkpoint_callback_config in (
                    self.config.callbacks.model_checkpoints
            ):
                model_checkpoint_callback_config_copy = OmegaConf.create(
                    OmegaConf.to_container(
                        model_checkpoint_callback_config,
                        resolve=True
                    )
                )
                model_checkpoint_callback_config_copy['filename'] = (
                    model_checkpoint_callback_config_copy['filename'].replace(
                        "exp=X-ver=Y-df=Z",
                        "exp={}-ver={}-df={}".format(
                            self.experiment_id,
                            self.experiment_version_id,
                            self.datafold_id
                        )
                    )
                )
                trainer_callbacks.append(ModelCheckpoint(
                    dirpath=(
                        f"{self.experiment_version_dir_path}"
                        f"/datafold_{self.datafold_id}/models"
                    ),
                    verbose=False,
                    **model_checkpoint_callback_config_copy
                ))
        if self.config.enable_model_early_stopping:
            for model_early_stopping_config in (
                    self.config.callbacks.model_early_stoppings
            ):
                trainer_callbacks.append(
                    EarlyStopping(**model_early_stopping_config)
                )

        return trainer_callbacks

    def _get_model_trainer_loggers(self):
        trainer_loggers = []
        if self.config.enable_logging:
            csv_logger = CSVLogger(
                datafold_id=self.datafold_id,
                version=self.experiment_version_id,
                name="",
                save_dir=self.experiment_dir_path
            )
            trainer_loggers.append(csv_logger)
        return trainer_loggers
