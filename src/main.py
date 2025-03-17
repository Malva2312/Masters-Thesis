import hydra
from modules.Manager import DataLoaderManager, TrainerManager, DataInfo

class MainApplication:
    def __init__(self):
        hydra.initialize(config_path='./config', version_base=None)
        self.config = hydra.compose(
            config_name="config", 
            overrides=[
                "data/preprocessed/loader=lidc_idri_preprocessed_data_loader_jn_demo",
                "metadata/preprocessed=lidc_idri_preprocessed_metadata_jn_demo"
            ]
        )
        self.dataloader_manager = DataLoaderManager(self.config)
        self.trainer_manager = TrainerManager(self.config, self.dataloader_manager)

    def run(self):
        print("\n-------------------------------------- Demonstrating the K-fold data loader ---------------------------------------\n")
        self.dataloader_manager.setup_k_fold_dataloader()
        data_loaders_by_subset = self.dataloader_manager.get_data_loaders_by_subset()
        for subset_type in ["train", "validation", "test"]:
            print(f"Subset type: {subset_type.title()}")
            for fold_index in range(self.config.data.preprocessed.loader.number_of_k_folds):
                print(f"    Fold index: {fold_index + 1}")
                for batch_index, (data, label) in enumerate(iter(data_loaders_by_subset[subset_type][fold_index]), 1):
                    data_info = DataInfo(batch_index=batch_index, data=data, label=label)
                    self.dataloader_manager.print_loaded_data_info(data_info, k_fold_data_loaders=True)

        print("\n-------------------------------------- Demonstrating the Training ---------------------------------------\n")
        self.trainer_manager.setup_model()
        self.trainer_manager.train_model()
        self.trainer_manager.test_model()

if __name__ == "__main__":
    app = MainApplication()
    app.run()
