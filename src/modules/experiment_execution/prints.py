class ExperimentExecutionPrints:

    @staticmethod
    def datafold_start(datafold_id):
        print(f'\nRunning model pipeline for data fold {datafold_id}...')

    @staticmethod
    def datafold_end(datafold_id):
        print(f'...model pipeline for data fold {datafold_id} has run!')

    @staticmethod
    def experiment_version_start(
            experiment_id,
            hyperparameter_combination_index
    ):
        print(
            "\n" + "*" * 50 +
            "  Experiment {} | Version {} ".format(
                experiment_id,
                hyperparameter_combination_index
            ) +
            "*" * 50
        )