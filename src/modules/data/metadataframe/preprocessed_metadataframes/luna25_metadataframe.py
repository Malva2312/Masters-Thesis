import pandas


class LUNA25PreprocessedMetaDataFrame:
    def __init__(self, config, experiment_execution_paths):
        self.config = config

        self.lung_nodule_metadataframe = pandas.read_csv(
            filepath_or_buffer="{}/protocol_{}".format(
                experiment_execution_paths.preprocessed_data_dir_path,
                self.config.data_preprocessing_protocol_number
            ) + "/metadata_csvs/lung_nodule_metadata.csv"
        )
        self._apply_lung_nodule_metadataframe_transformations()

    def get_lung_nodule_metadataframe(self):
        return self.lung_nodule_metadataframe

    def _apply_lung_nodule_metadataframe_transformations(self):
        self.lung_nodule_metadataframe['file_name'] = \
            self.lung_nodule_metadataframe.apply(
            lambda row:
                f"P{row['patient_id']}-"
                f"S{row['series_instance_uid'][-5:]}-"
                f"N{str(row['nodule_id']).zfill(2)}",
            axis=1
        )
        self.lung_nodule_metadataframe = \
            self.lung_nodule_metadataframe[['file_name', 'label']]
        self.lung_nodule_metadataframe['label'] = \
            self.lung_nodule_metadataframe['label'].astype(int)
