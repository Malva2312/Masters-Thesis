import hydra
from modules.Manager import DataLoaderManager, TrainerManager, DataInfo
import argparse
import yaml

class MainApplication:
    def __init__(self, args):
        hydra.initialize(config_path='./config', version_base=None)
        if args.demo_mode:
            dataloader_override = "data/preprocessed/loader=lidc_idri_preprocessed_data_loader_jn_demo"
            metadata_override = "metadata/preprocessed=lidc_idri_preprocessed_metadata_jn_demo"
            print("Using demo dataloader")
        else:
            dataloader_override = "data/preprocessed/loader=lidc_idri_preprocessed_data_loader"
            metadata_override = "metadata/preprocessed=lidc_idri_preprocessed_metadata"
            print("Using non-demo dataloader")

        self.config = hydra.compose(
            config_name="config", 
            overrides=[
                dataloader_override,
                metadata_override
            ]
        )
        self.dataloader_manager = DataLoaderManager(self.config)
        self.trainer_manager = TrainerManager(
            self.config, 
            self.dataloader_manager, 
            protocol=args.protocol, 
            protocol_params=args.protocol_params
        )

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
    parser = argparse.ArgumentParser(description="Main application for training lung nodule classifiers.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--demo_mode", action="store_true", help="Use non-demo dataloader if set, otherwise use demo dataloader.")
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config_data = yaml.safe_load(file)

    args.protocol = config_data.get("protocol", None)
    args.protocol_params = config_data.get("protocol_params", {})

    app = MainApplication(args)
    app.run()
