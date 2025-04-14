from datetime import datetime
from os.path import exists
import pandas
import textwrap


class ExperimentExecutionInfo:
    def __init__(
            self,
            config,
            experiment_execution_ids,
            experiment_execution_paths
    ):
        self.config = config

        self.csv_file_path = (
            f"{experiment_execution_paths.python_project_dir_path}"
            f"/experiment_results/experiments_info.csv"
        )
        self.experiment_id = experiment_execution_ids.experiment_id
        self.md_file_path = (
            f"{experiment_execution_paths.experiment_dir_path}"
            f"/experiment_info.md"
        )
        if exists(self.csv_file_path):
            self.dataframe = pandas.read_csv(self.csv_file_path)
            self.dataframe_data = self.dataframe.to_dict(orient="records")
        else:
            self.dataframe_data = []

        self.check_required_info()

    def check_required_info(self):
        missing_info = [
            key for key in ["who", "what"]
            if self.config.get(key) is None
        ]
        if missing_info:
            raise ValueError(
                f"Error: Missing values for 'who' and 'what' "
                f"in config/experiment_execution/info/info.yaml"
            )

    def set_dataframe(self):
        self.dataframe_data.append({
            "Experiment ID": self.experiment_id,
            "Who ran it?": self.config.who,
            "When was it run?": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "What was run?": self.config.what
        })

        self.dataframe = pandas.DataFrame(self.dataframe_data)
        self.dataframe.drop_duplicates(
            subset="Experiment ID",
            keep="last",
            inplace=True
        )

    def save_as_md(self):
        md_file_content = textwrap.dedent(f"""\
            Experiment ID: {self.experiment_id}
            Who ran it: {self.config.who}
            When was it run: {datetime.now().strftime("%Y-%m-%d %H:%M")}
            What was run: {self.config.what}
        """)

        with open(self.md_file_path, "w") as file:
            file.write(md_file_content)

    def save_dataframe_as_csv(self):
        self.dataframe.to_csv(
            path_or_buf=self.csv_file_path,
            encoding='utf-8',
            index=False
        )
