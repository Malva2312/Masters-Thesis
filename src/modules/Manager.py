from os.path import abspath, dirname, join
import sys
from lightning import Trainer



sys.path.append(abspath(join(dirname(__file__), "../data_loading/")))
from data_loading.src.modules.data.dataloader.preprocessed_data_loader import LIDCIDRIPreprocessedKFoldDataLoader
from data_loading.src.modules.data.metadata import LIDCIDRIPreprocessedMetaData
from data_loading.src.modules.utils.paths import PYTHON_PROJECT_DIR_PATH

from modules.LungNoduleClassifier import LungNoduleClassifier
from modules.protocols.ProtocolEfficientNet import ProtocolEfficientNet

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
        print(f"{space}    Batch index: {data_info.batch_index}")
        print(f"{space}        Data (Lung nodule CT image):")
        print(f"{space}         - Type: {type(data_info.data['input_image']).__name__}")
        print(f"{space}         - Shape: {data_info.data['input_image'].shape}")
        print(f"{space}         - Min/max values: {data_info.data['input_image'].min()}/{data_info.data['input_image'].max()}")
        print(f"{space}        Label (Mean lung nodule malignancy)")
        print(f"{space}         - Type: {type(data_info.label['lnm']['mean']).__name__}")
        print(f"{space}         - Shape: {data_info.label['lnm']['mean'].shape}")
        print(f"{space}         - Min/max values: {data_info.label['lnm']['mean'].min()}/{data_info.label['lnm']['mean'].max()}")

class TrainerManager:
    def __init__(self, config, dataloader_manager):
        self.config = config
        self.dataloader_manager = dataloader_manager
        self.trainer = Trainer(limit_train_batches=100, max_epochs=1)
        self.autoencoder = None

    def setup_model(self):
        #self.autoencoder = LungNoduleClassifier()
        self.autoencoder = ProtocolEfficientNet(num_classes=2, num_channels=1)

    def train_model(self):
        self.trainer.fit(model=self.autoencoder, train_dataloaders=self.dataloader_manager.get_data_loaders_by_subset()["train"][0])

    def test_model(self):
        encoder = self.autoencoder.encoder
        encoder.eval()
