from plotly.subplots import make_subplots
from PIL import ImageColor
import math
import numpy
import os
import pandas
import plotly.graph_objects


class ModelTrainingPerformanceMetricsFigure:
    def __init__(
            self,
            config,
            experiment_execution_ids,
            experiment_execution_paths
    ):
        self.config = config

        self.experiment_id = experiment_execution_ids.experiment_id
        self.experiment_version_id = \
            experiment_execution_ids.experiment_version_id
        self.experiment_version_dir_path = \
            experiment_execution_paths.experiment_version_dir_path
        self.figure = None

    def add_loss_related_traces(
            self,
            performance_metrics_dataframe_column_names,
            trace_column_number,
            trace_legend_group_title,
            trace_line_colours,
            trace_names
    ):
        performance_metrics_dataframes = \
            self.get_performance_metrics_dataframes(
                performance_metrics_dataframe_column_names
            )
        for df_column_name, trace_name, trace_line_colour in zip(
                performance_metrics_dataframe_column_names,
                trace_names,
                trace_line_colours
        ):
            for data_fold_id in range(1, 6):
                self.figure.add_trace(
                    plotly.graph_objects.Scatter(
                        y=performance_metrics_dataframes[
                            f'datafold_{data_fold_id}'
                        ][df_column_name],
                        name=trace_name,
                        hovertemplate="%{x}, %{y:.2f}",
                        mode='lines',
                        line=dict(color=trace_line_colour, width=2),
                        legendgrouptitle_text=trace_legend_group_title,
                        legendgroup=trace_legend_group_title,
                        showlegend= True if data_fold_id == 1 else False
                    ),
                    row=data_fold_id,
                    col=trace_column_number
                )

            metric_lists = [
                performance_metrics_dataframes[f'datafold_{data_fold_id}'][
                    df_column_name
                ].to_list() for data_fold_id in range(1, 6)
            ]
            x_axis_max_value = max([
                len(metric_list) for metric_list in metric_lists
            ])
            metric_padded_lists = [
                numpy.pad(
                    metric_list,
                    (0, x_axis_max_value - len(metric_list)),
                    constant_values=numpy.nan
                )
                for metric_list in metric_lists
            ]
            overall_metric_means = numpy.nanmean(metric_padded_lists, axis=0)
            overall_metric_stds = numpy.nanstd(metric_padded_lists, axis=0)

            x = list(range(len(overall_metric_means)))
            self.figure.add_trace(
                plotly.graph_objects.Scatter(
                    y=overall_metric_means,
                    name=trace_name,
                    mode='lines',
                    hovertemplate="%{x}, %{y:.2f}",
                    line=dict(color=trace_line_colour, width=2),
                    legendgroup=trace_legend_group_title,
                    showlegend=False
                ),
                row=6,
                col=trace_column_number
            )
            self.figure.add_trace(
                plotly.graph_objects.Scatter(
                    y=numpy.concatenate([
                        overall_metric_means - overall_metric_stds,
                        (overall_metric_means + overall_metric_stds)[::-1]
                    ]),
                    x=numpy.concatenate([x, x[::-1]]),
                    mode='lines',
                    hovertemplate="%{x}, %{y:.2f}",
                    line=dict(color=trace_line_colour, width=0.2),
                    fill='toself',
                    fillcolor="rgba{}".format(
                        ImageColor.getrgb(trace_line_colour) + (0.2,)
                    ),
                    legendgroup=trace_legend_group_title,
                    showlegend=False
                ),
                row=6,
                col=trace_column_number
            )

    def add_validation_performance_metrics_traces(
            self,
            performance_metrics_dataframe_column_names,
            trace_column_number,
            trace_legend_group_title,
            trace_line_colours,
            trace_names
    ):
        performance_metrics_dataframes = \
            self.get_performance_metrics_dataframes(
                performance_metrics_dataframe_column_names
            )
        for df_column_name, trace_name, trace_line_colour in zip(
                performance_metrics_dataframe_column_names,
                trace_names,
                trace_line_colours
        ):
            for data_fold_id in range(1, 6):
                self.figure.add_trace(
                    plotly.graph_objects.Scatter(
                        y=performance_metrics_dataframes[
                            f'datafold_{data_fold_id}'
                        ][df_column_name],
                        name=trace_name,
                        hovertemplate="%{x}, %{y:.2f}",
                        mode='lines',
                        line=dict(color=trace_line_colour, width=2),
                        legendgrouptitle_text=trace_legend_group_title,
                        legendgroup=trace_legend_group_title,
                        showlegend=True if data_fold_id == 1 else False
                    ),
                    row=data_fold_id,
                    col=trace_column_number
                )

            metric_lists = [
                performance_metrics_dataframes[f'datafold_{data_fold_id}'][
                    df_column_name
                ].to_list() for data_fold_id in range(1, 6)
            ]
            x_axis_max_value = max([
                len(metric_list) for metric_list in metric_lists
            ])
            metric_padded_lists = [
                numpy.pad(
                    metric_list,
                    (0, x_axis_max_value - len(metric_list)),
                    constant_values=numpy.nan
                )
                for metric_list in metric_lists
            ]
            overall_metric_means = numpy.nanmean(metric_padded_lists, axis=0)
            overall_metric_stds = numpy.nanstd(metric_padded_lists, axis=0)

            x = list(range(len(overall_metric_means)))
            self.figure.add_trace(
                plotly.graph_objects.Scatter(
                    y=overall_metric_means,
                    name=trace_name,
                    mode='lines',
                    hovertemplate="%{x}, %{y:.2f}",
                    line=dict(color=trace_line_colour, width=2),
                    legendgroup=trace_legend_group_title,
                    showlegend=False
                ),
                row=6,
                col=trace_column_number
            )
            self.figure.add_trace(
                plotly.graph_objects.Scatter(
                    y=numpy.concatenate([
                        overall_metric_means - overall_metric_stds,
                        (overall_metric_means + overall_metric_stds)[::-1]
                    ]),
                    x=numpy.concatenate([x, x[::-1]]),
                    mode='lines',
                    hovertemplate="%{x}, %{y:.2f}",
                    line=dict(color=trace_line_colour, width=0.2),
                    fill='toself',
                    fillcolor="rgba{}".format(
                        ImageColor.getrgb(trace_line_colour) + (0.2,)
                    ),
                    legendgroup=trace_legend_group_title,
                    showlegend=False
                ),
                row=6,
                col=trace_column_number
            )

    def get_axis_max_values(self):
        train_and_validation_loss_dataframes = \
            self.get_performance_metrics_dataframes(
                column_names=["train_loss", "val_loss"]
            )
        validation_performance_metric_dataframes = \
            self.get_performance_metrics_dataframes(
                column_names=[
                    "val_auroc",
                    "val_accuracy",
                    "val_precision",
                    "val_recall"
                ]
            )

        # x_axis_max_value = max([len(
        #     train_and_validation_loss_dataframes[f'datafold_{data_fold_id}'][
        #         'train_loss'
        #     ].iloc[:-1].to_list()) for data_fold_id in range(1, 6)
        # ])
        x_axis_max_value = self.config.maximum_number_of_epochs

        train_and_validation_loss_y_axis_max_values = {
            y_axis_idx: math.ceil(
                train_and_validation_loss_dataframes[
                    f'datafold_{data_fold_id}'
                ].max().max()
            ) for data_fold_id, y_axis_idx in zip(
                range(1, 6),
                range(1, 13, 2)
            )
        }
        train_and_validation_loss_y_axis_max_values[11] = max(
            train_and_validation_loss_y_axis_max_values.values()
        )
        validation_performance_metric_y_axis_max_values = {
            y_axis_idx: math.ceil(
                validation_performance_metric_dataframes[
                    f'datafold_{data_fold_id}'
                ].max().max()
            ) for data_fold_id, y_axis_idx in zip(
                range(1, 6),
                range(2, 13, 2)
            )
        }
        validation_performance_metric_y_axis_max_values[12] = max(
            validation_performance_metric_y_axis_max_values.values()
        )

        y_axis_max_values = {
            'x': x_axis_max_value,
            'y': {
                **train_and_validation_loss_y_axis_max_values,
                **validation_performance_metric_y_axis_max_values,
                # **validation_lnva_accuracy_y_axis_max_values,
                # **validation_lnm_accuracy_y_axis_max_values
            }
        }
        return y_axis_max_values

    def get_performance_metrics_dataframes(self, column_names):
        performance_metrics_dataframes = dict()
        for data_fold_id in range(1, 6):
            performance_metrics_file_path = (
                f"{self.experiment_version_dir_path}"
                 f"/datafold_{data_fold_id}/metrics.csv"
            )
            if os.path.exists(performance_metrics_file_path):
                performance_metrics_dataframe = \
                    pandas.read_csv(performance_metrics_file_path)
                performance_metrics_dataframes[f'datafold_{data_fold_id}'] = (
                    performance_metrics_dataframe[
                        ['epoch', *column_names]
                    ].set_index('epoch').groupby(level="epoch")
                        .mean().sort_values(by='epoch')
                )
                performance_metrics_dataframes[f'datafold_{data_fold_id}'] = \
                    performance_metrics_dataframes \
                        [f'datafold_{data_fold_id}'].dropna()
        return performance_metrics_dataframes

    def plot(self):
        self.figure.show()

    def save_image(self):
        self.figure.write_image(
            f"{self.experiment_version_dir_path}"
            f"/model_training_performance_metrics.png"
        )

    def set(self):
        self.figure = make_subplots(
            rows=6,
            cols=2,
            vertical_spacing=0.075,
            subplot_titles=(
                "<b>Fig. 1 - Loss X Epochs</b>",
                "<b>Fig. 2 - LNM score prediction performance metrics X Epochs</b>"
            )
        )
        self.add_loss_related_traces(
            performance_metrics_dataframe_column_names=
                ["train_loss", "val_loss"],
            trace_column_number=1,
            trace_legend_group_title="<b>Figure 1</b>",
            trace_line_colours=["orange", "steelblue"],
            trace_names = ["Train", "Validation"]
        )
        self.add_validation_performance_metrics_traces(
            performance_metrics_dataframe_column_names=
                ["val_auroc", "val_accuracy", "val_precision", "val_recall"],
            trace_column_number=2,
            trace_legend_group_title="<b>Figure 2</b>",
            trace_line_colours=["black", "magenta", "green", "red"],
            trace_names=["AUC", "Accuracy", "Precision", "Sensitivity"]
        )
        self.update_layout()

    def update_layout(self):
        figure_title = (
            f"<span style='font-size:30'><b>Model training "
             f"performance metrics</b></span>"
        )
        figure_subtitle = (
            f"<span style='font-size:15'>"
            f"Experiment ID <b>{self.experiment_id}"
            f"</b>        Version ID <b>"
            f"{self.experiment_version_id}</b></span>"
        )
        axis_max_values = self.get_axis_max_values()
        self.figure.update_layout(
            height=975,
            hovermode="x",
            legend=dict(
                groupclick="toggleitem",
                font=dict(size=14),
                grouptitlefont=dict(size=15),
                tracegroupgap=25
            ),
            margin=dict(t=180, l=200, r=300),
            title=dict(
                text=f"{figure_title}<br><br>{figure_subtitle}",
                x=0.5,
                y=0.94,
                font=dict(size=15)
            ),
            width=1500,
            **{
                f'xaxis{xaxis_idx}': dict(
                    dtick=axis_max_values['x'] // 5,
                    range=[0, axis_max_values['x'] - 1]
                )
                for xaxis_idx in range(1, 13)
            },
            yaxis=dict(
                side="left",
                dtick=axis_max_values['x'] // 4,
                range=[0, axis_max_values['y'][1]]
            ),
            yaxis11=dict(
                side="left",
                dtick=axis_max_values['x'] // 4,
                title=(
                    "<b>Data fold<br><span style='font"
                    "-size:11.2'>mean ± std</span></b>"
                ),
                range=[0, axis_max_values['y'][11]]
            ),
            **{
                f'yaxis{yaxis_idx}': dict(
                    side="left",
                    dtick=axis_max_values['x'] // 4,
                    title=f"<b>Data fold {data_fold_id}</b>",
                    range=[0, axis_max_values['y'][yaxis_idx]]
                ) for data_fold_id, yaxis_idx in zip(
                    range(1, 6), range(1, 13, 2)
                )
            },
            **{
                f'yaxis{yaxis_idx}': dict(
                    side="left",
                    dtick=axis_max_values['x'] // 4,
                    range=[0, 1]
                ) for yaxis_idx in range(2, 13, 2)
            }
        )
        self.figure.update_annotations(font_size=12, yshift=12)
