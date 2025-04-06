from os.path import abspath, dirname, join
import sys
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger



sys.path.append(abspath(join(dirname(__file__), "../data_loading/")))
from data_loading.src.modules.data.dataloader.preprocessed_data_loader import LIDCIDRIPreprocessedKFoldDataLoader
from data_loading.src.modules.data.metadata import LIDCIDRIPreprocessedMetaData
from data_loading.src.modules.utils.paths import PYTHON_PROJECT_DIR_PATH

# from modules.LungNoduleClassifier import LungNoduleClassifier # Example protoco 
from modules.protocols.ProtocolEfficientNet import ProtocolEfficientNet
from modules.protocols.ProtocolRadiomics import ProtocolRadiomics

class DataInfo:
    def __init__(self, batch_index, data, label):
        self.batch_index = batch_index
        self.data = data
        self.label = label

class DataLoaderManager:
    def __init__(self, config):
        self.config = config
        self.metadata = LIDCIDRIPreprocessedMetaData(config=config.metadata.preprocessed)
        self.dataloader = None

    def setup_k_fold_dataloader(self):
        self.config.data.preprocessed.loader.number_of_k_folds = 5
        self.config.data.preprocessed.loader.test_fraction_of_entire_dataset = None
        self.dataloader = LIDCIDRIPreprocessedKFoldDataLoader(
            config=self.config.data.preprocessed.loader, 
            lung_nodule_image_metadataframe=self.metadata.get_lung_nodule_image_metadataframe()
        )

    def get_data_loaders_by_subset(self):
        return self.dataloader.get_data_loaders_by_subset()

    def print_loaded_data_info(self, data_info, k_fold_data_loaders=False, load_mask=False):
        space = "    " if k_fold_data_loaders else ""
        # print(f"{space}    Batch index: {data_info.batch_index}")
        # print(f"{space}        Data (Lung nodule CT image):")
        # print(f"{space}         - Type: {type(data_info.data['input_image']).__name__}")
        # print(f"{space}         - Shape: {data_info.data['input_image'].shape}")
        # print(f"{space}         - Min/max values: {data_info.data['input_image'].min()}/{data_info.data['input_image'].max()}")
        # print(f"{space}        Label (Mean lung nodule malignancy)")
        # print(f"{space}         - Type: {type(data_info.label['lnm']['mean']).__name__}")
        # print(f"{space}         - Shape: {data_info.label['lnm']['mean'].shape}")
        # print(f"{space}         - Min/max values: {data_info.label['lnm']['mean'].min()}/{data_info.label['lnm']['mean'].max()}")

class TrainerManager:
    def __init__(self, config, dataloader_manager, protocol, protocol_params):
        self.config = config
        self.dataloader_manager = dataloader_manager
        self.protocol = protocol
        self.protocol_params = protocol_params
        
        logger = TensorBoardLogger(save_dir="lightning_logs/")
        self.trainer = Trainer(logger=logger, limit_train_batches=100, max_epochs=1)
        self.autoencoder = None

        print(f"Protocol: {self.protocol}")
        print(f"Protocol parameters: {self.protocol_params}")


    def setup_model(self):
        if self.protocol == "ProtocolEfficientNet":
            self.autoencoder = ProtocolEfficientNet(**self.protocol_params)
        elif self.protocol == "ProtocolRadiomics":
            self.autoencoder = ProtocolRadiomics(**self.protocol_params)
        else:
            raise ValueError(f"Unknown protocol: {self.protocol}")

    def train_model(self):
        self.trainer.fit(model=self.autoencoder, train_dataloaders=self.dataloader_manager.get_data_loaders_by_subset()["train"][0])

    def test_model(self):
        test_dataloader = self.dataloader_manager.get_data_loaders_by_subset()["test"][0]
        test_results = self.trainer.test(model=self.autoencoder, dataloaders=test_dataloader)
        print("Test Results:", test_results)