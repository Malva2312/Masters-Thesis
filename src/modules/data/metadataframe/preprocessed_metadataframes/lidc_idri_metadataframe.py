import ast
import numpy
import pandas
import statistics


class LIDCIDRIPreprocessedMetaDataFrame:
    def __init__(self, config, experiment_execution_paths):
        self.config = config

        self.lung_nodule_metadataframe = pandas.read_csv(
            filepath_or_buffer="{}/protocol_{}".format(
                experiment_execution_paths.preprocessed_data_dir_path,
                self.config.data_preprocessing_protocol_number
            ) + "/metadata_csvs/lung_nodule_image_metadata.csv"
        )
        self._apply_lung_nodule_metadataframe_transformations()

    # def get_file_names(self):
    #     file_names = self.lung_nodule_metadataframe['file_name'].tolist()
    #     return file_names

    def get_lung_nodule_metadataframe(self):
        return self.lung_nodule_metadataframe

    # def get_visual_attribute_score_means_dataframe(self):
    #     visual_attribute_score_means_dataframe = self.lung_nodule_metadataframe.copy()
    #     filtered_df = self.lung_nodule_metadataframe.loc[:,
    #         'mean' in self.lung_nodule_metadataframe.columns
    #         & 'file_name' in self.lung_nodule_metadataframe.columns]
    #     return visual_attribute_score_means_dataframe

    def _apply_lung_nodule_metadataframe_transformations(self):
        # insert nodule file names
        self.lung_nodule_metadataframe.insert(
            loc=0, column='file_name', value=(
                self.lung_nodule_metadataframe['Patient ID']
                + "-N" + self.lung_nodule_metadataframe['Nodule ID']
                    .astype(str).str.zfill(2)
            )
        )

        # set up nodule malignancy columns
        self.lung_nodule_metadataframe['Nodule Malignancy'] = \
            self.lung_nodule_metadataframe['Nodule Malignancy'] \
                .apply(ast.literal_eval)
        self.lung_nodule_metadataframe.insert(
            loc=self.lung_nodule_metadataframe.columns
                .get_loc('Nodule Malignancy') + 1,
            column=f"Mean Nodule Malignancy",
            value=self.lung_nodule_metadataframe['Nodule Malignancy']
                .apply(numpy.mean)
        )
        self.lung_nodule_metadataframe.insert(
            loc=self.lung_nodule_metadataframe.columns
                .get_loc('Nodule Malignancy') + 2,
            column=f"Nodule Malignancy StD",
            value=self.lung_nodule_metadataframe['Nodule Malignancy']
                .apply(numpy.std)
        )

        # set up nodule visual attribute columns
        for lnva_name in self.config.lnva.names:
            self.lung_nodule_metadataframe[
                f'Nodule {lnva_name.replace("_", " ").title()}'
            ] = self.lung_nodule_metadataframe[
                f'Nodule {lnva_name.replace("_", " ").title()}'
            ].apply(ast.literal_eval)


            statistical_operation = numpy.mean
            if self.config.use_mode_for_internal_structure_and_calcification:
                if lnva_name in ["internal_structure", "calcification"]:
                    print(f"using mode for {lnva_name}")
                    statistical_operation = statistics.mode
            self.lung_nodule_metadataframe.insert(
                loc=self.lung_nodule_metadataframe.columns.get_loc(
                    f'Nodule {lnva_name.replace("_", " ").title()}'
                ) + 1,
                column=f'Mean Nodule {lnva_name.replace("_", " ").title()}',
                value=self.lung_nodule_metadataframe[
                    f'Nodule {lnva_name.replace("_", " ").title()}'
                ].apply(statistical_operation))

        # filter nodules that have been labeled by at least three radiologists
        self.lung_nodule_metadataframe = self.lung_nodule_metadataframe[
            self.lung_nodule_metadataframe['Nodule Malignancy']
                .apply(lambda x: len(x) >= 3)
        ]

        # filter nodules with mean nodule malignancy score != 3
        self.lung_nodule_metadataframe = self.lung_nodule_metadataframe[
            self.lung_nodule_metadataframe[f"Mean Nodule Malignancy"] != 3]

        # reset index due to applied filters
        self.lung_nodule_metadataframe.reset_index(drop=True, inplace=True)

        self.lung_nodule_metadataframe.insert(
            loc=len(self.lung_nodule_metadataframe.columns),
            column=f"label",
            value=(
                self.lung_nodule_metadataframe['Mean Nodule Malignancy'] > 3
            ).astype(int)
        )
