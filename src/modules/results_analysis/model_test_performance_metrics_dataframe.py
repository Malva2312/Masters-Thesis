import numpy
import os
import pandas


class ModelTestPerformanceMetricsDataframe:
    def __init__(
            self,
            config,
            experiment_execution_ids,
            experiment_execution_paths
    ):
        self.config = config

        self.datafold_header_column_names = [
            "Experiment version ID",
            "Monitored variable",
            "AUC",
            "Accuracy",
            "Precision",
            "Sensitivity"
        ]
        self.datafold_test_metric_names = [
            "test_auroc",
            "test_accuracy",
            "test_precision",
            "test_recall",

        ]
        self.dataframe = pandas.DataFrame(
            columns=self.datafold_header_column_names
        )
        self.experiment_id = experiment_execution_ids.experiment_id
        self.experiment_dir_path = \
            experiment_execution_paths.experiment_dir_path
        self.experiment_version_ids = [
            folder.split("_")[1] for folder in
            os.listdir(self.experiment_dir_path)
            if os.path.isdir(os.path.join(self.experiment_dir_path, folder))
        ]

    def export_as_html(self):
        dataframe = self.dataframe.copy()
        styler = self._change_style(dataframe)
        html_dataframe = styler.to_html(escape=False, encoding="utf-8")
        html_dataframe = html_dataframe.replace("±", "&plusmn;")
        html_dataframe = html_dataframe + (
            "<p>Cell numeric values represent the mean &plusmn; std "
            "across all 5 folds of the model test performance metrics. "
            "The <span style='background-color: limegreen;'><b>"
            "&nbsp;green&nbsp;</b></span> background "
            "highlights the highest mean in the column.</p>"
        )

        with open(
                f"{self.experiment_dir_path}"
                f"/Model_test_performance_metrics.html",
                "w",
                encoding="utf-8"
        ) as file:
            file.write(html_dataframe)

    def set_dataframe(self):
        for experiment_version_id in self.experiment_version_ids:
            experiment_version_dir = \
                f"{self.experiment_dir_path}/version_{experiment_version_id}"
            if os.path.isdir(experiment_version_dir):
                for (
                        monitored_variable_index,
                        monitored_variable_name
                ) in enumerate(self.config.monitored_variables, start=1):
                    performance_metric_values = []
                    for data_fold_id in range(1, 6):
                        metrics_path = (
                            f"{experiment_version_dir}/"
                            f"datafold_{data_fold_id}/metrics.csv"
                        )
                        if os.path.isfile(metrics_path):
                            metrics_dataframe = pandas.read_csv(metrics_path)
                            if pandas.Index(
                                self.datafold_test_metric_names
                            ).isin(metrics_dataframe.columns).all():
                                performance_metric_values.append(
                                    metrics_dataframe[
                                        self.datafold_test_metric_names
                                    ].dropna().iloc[
                                        monitored_variable_index - 1
                                    ].tolist()
                                )
                    if performance_metric_values:
                        performance_metric_mean_values = numpy.mean(
                            performance_metric_values, axis=0)
                        performance_metric_values_std = numpy.std(
                            performance_metric_values, axis=0)
                        if (
                                isinstance(
                                    performance_metric_mean_values,
                                    numpy.ndarray
                                ) and len(
                                    performance_metric_mean_values
                                ) == len(
                                    self.datafold_test_metric_names
                                )
                        ):
                            performance_metric_mean_values = \
                                numpy.array(performance_metric_mean_values)
                            performance_metric_mean_values[9:] *= 100
                            performance_metric_mean_values = \
                                list(performance_metric_mean_values)

                            performance_metric_values_std = \
                                numpy.array(performance_metric_values_std)
                            performance_metric_values_std[9:] *= 100
                            performance_metric_values_std = \
                                list(performance_metric_values_std)

                            performance_metric_values = [
                                f"{round(performance_metric_mean_value, 2)}"
                                .ljust(4, '0') + " ± " +
                                f"{round(performance_metric_value_std, 2)}"
                                .ljust(4, '0')
                                for
                                    performance_metric_mean_value,
                                    performance_metric_value_std
                                in zip(
                                    performance_metric_mean_values,
                                    performance_metric_values_std
                                )
                            ]

                            self.dataframe.loc[
                                len(self.dataframe),
                                self.dataframe.columns
                            ] = [
                                experiment_version_id,
                                monitored_variable_name
                            ] + performance_metric_values

    def _change_style(self, dataframe):

        def apply_striped_background(row):
            if (row.name // 3) % 2 == 0:
                return ['background-color: lightgray'] * len(row)
            else:
                return ['background-color: white'] * len(row)

        def highlight_column_best_values(
                dataframe_column_with_mean_and_std_values
        ):
            dataframe_column_with_mean_values = \
                dataframe_column_with_mean_and_std_values.apply(
                    lambda x: float(x.split(' ± ')[0])
                )
            dataframe_column_max_value = \
                dataframe_column_with_mean_values.max()
            return [
                'background-color: limegreen; font-weight: bold'
                if float(
                    cell_value.split(' ± ')[0]
                ) == dataframe_column_max_value else ''
                for cell_value in dataframe_column_with_mean_and_std_values
            ]

        styler = dataframe.style
        styler = styler.apply(apply_striped_background, axis=1)
        styler = styler.apply(
            highlight_column_best_values,
            subset=["AUC", "Accuracy", "Precision", "Sensitivity"],
            axis=0
        )

        experiment_version_ids_range = f"{self.experiment_version_ids[0]}"
        if len(self.experiment_version_ids) > 1:
            experiment_version_ids_range += \
                f" to {self.experiment_version_ids[-1]}"
        styler.set_caption(
            "<span style='font-size:30px;'><b>"
            "Model test performance metrics</b></span>" +
            f"<span style='font-size:16px;'><br style='line-height: 2.5;'>"
            f"Experiment ID: <b>{self.experiment_id}</b>" + 10 * "&nbsp;" +
            f"Version: <b>{experiment_version_ids_range}</b></span>"
        )
        styler = styler.set_table_styles([
            {'selector': 'th', 'props': [
                ('font-size', '13.5px'),
                ('padding-left', '7.5px'),
                ('padding-right', '7.5px'),
                ('text-align', 'left'),
                ('white-space', 'nowrap')
            ]},
            {'selector': 'td', 'props': [
                ('font-size', '13.5px'),
                ('padding-left', '7.5px'),
                ('padding-right', '7.5px'),
                ('white-space', 'nowrap')
            ]},
            {'selector': 'caption', 'props': [
                ('margin-bottom', '30px'),
                ('margin-top', '30px')
            ]}

        ], overwrite=False)
        return styler