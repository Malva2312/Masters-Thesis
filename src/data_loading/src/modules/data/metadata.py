import ast
import numpy
import pandas
import statistics

from src.modules.utils.paths import get_preprocessed_data_dir_path


class LIDCIDRIPreprocessedMetaData:
    def __init__(self, config):
        self.config = config

        self._lung_nodule_image_metadataframe = None
        self._set_lung_nodule_image_metadataframe()

    def get_file_names(self):
        file_names = self._lung_nodule_image_metadataframe['file_name'].tolist()
        return file_names

    def get_lung_nodule_image_metadataframe(self):
        return self._lung_nodule_image_metadataframe

    def get_visual_attribute_score_means_dataframe(self):
        visual_attribute_score_means_dataframe = \
            self._lung_nodule_image_metadataframe.copy()
        filtered_df = self._lung_nodule_image_metadataframe.loc[:,
            'mean' in self._lung_nodule_image_metadataframe.columns
            & 'file_name' in self._lung_nodule_image_metadataframe.columns]
        return visual_attribute_score_means_dataframe

    def _set_lung_nodule_image_metadataframe(self):
        self._lung_nodule_image_metadataframe = pandas.read_csv(
            filepath_or_buffer=
                "{}/protocol_{}/metadata_csvs/lung_nodule_image_metadata.csv"
                    .format(
                    get_preprocessed_data_dir_path(
                        data_storage_source=self.config.metadata_storage_source
                    ),
                        self.config.data_preprocessing_protocol_number
                    )
        )

        # insert nodule file names
        self._lung_nodule_image_metadataframe.insert(
            loc=0, column='file_name', value=(
                self._lung_nodule_image_metadataframe['Patient ID']
                + "-N" + self._lung_nodule_image_metadataframe['Nodule ID']
                    .astype(str).str.zfill(2)
            )
        )

        # set up nodule malignancy columns
        self._lung_nodule_image_metadataframe['Nodule Malignancy'] = \
            self._lung_nodule_image_metadataframe['Nodule Malignancy'] \
                .apply(ast.literal_eval)
        self._lung_nodule_image_metadataframe.insert(
            loc=self._lung_nodule_image_metadataframe.columns \
                .get_loc('Nodule Malignancy') + 1,
            column=f"Mean Nodule Malignancy",
            value=self._lung_nodule_image_metadataframe['Nodule Malignancy'] \
                .apply(numpy.mean))
        self._lung_nodule_image_metadataframe.insert(
            loc=self._lung_nodule_image_metadataframe.columns \
                .get_loc('Nodule Malignancy') + 2,
            column=f"Nodule Malignancy StD",
            value=self._lung_nodule_image_metadataframe['Nodule Malignancy'] \
                .apply(numpy.std))

        # set up nodule visual attribute columns
        for lnva_name in self.config.lnva.names:
            self._lung_nodule_image_metadataframe[
                f'Nodule {lnva_name.replace("_", " ").title()}'
            ] = self._lung_nodule_image_metadataframe[
                f'Nodule {lnva_name.replace("_", " ").title()}'
            ].apply(ast.literal_eval)

            statistical_operation = numpy.mean
            if self.config.use_mode_for_internal_structure_and_calcification:
                if lnva_name in ["internal_structure", "calcification"]:
                    print(f"using mode for {lnva_name}")
                    statistical_operation = statistics.mode
            self._lung_nodule_image_metadataframe.insert(
                loc=self._lung_nodule_image_metadataframe.columns.get_loc(
                    f'Nodule {lnva_name.replace("_", " ").title()}'
                ) + 1,
                column=f'Mean Nodule {lnva_name.replace("_", " ").title()}',
                value=self._lung_nodule_image_metadataframe[
                    f'Nodule {lnva_name.replace("_", " ").title()}'
                ].apply(statistical_operation))
            statistical_operation = numpy.std
            self._lung_nodule_image_metadataframe.insert(
                loc=self._lung_nodule_image_metadataframe.columns.get_loc(
                    f'Mean Nodule {lnva_name.replace("_", " ").title()}'
                ) + 1,
                column=f'Nodule {lnva_name.replace("_", " ").title()} StD',
                value=self._lung_nodule_image_metadataframe[
                    f'Nodule {lnva_name.replace("_", " ").title()}'
                ].apply(statistical_operation))

        # filter nodules that have been labeled by at least three radiologists
        self._lung_nodule_image_metadataframe = \
            self._lung_nodule_image_metadataframe[
                self._lung_nodule_image_metadataframe['Nodule Malignancy'] \
                    .apply(lambda x: len(x) >= 3)
            ]

        # filter nodules with mean nodule malignancy score != 3
        self._lung_nodule_image_metadataframe = \
            self._lung_nodule_image_metadataframe[
                self._lung_nodule_image_metadataframe[
                    f"Mean Nodule Malignancy"
                ] != 3
            ]

        # reset index due to applied filters
        self._lung_nodule_image_metadataframe.reset_index(
            drop=True, inplace=True
        )
