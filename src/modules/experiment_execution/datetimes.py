from collections import defaultdict
import json


class ExperimentExecutionDatetimes:
    """A utility class for managing and saving event execution datetimes.

    This class provides functionality to store event names with their
    corresponding datetimes and save the data to a JSON file.

    Attributes:
        _dict (dict): A dictionary mapping event names to their datetimes as
            strings.

    Methods:
        add(event_name, datetime):
            Adds an event and its corresponding datetime to the dictionary.
        save():
            Saves the event datetimes dictionary to a JSON file.
    """

    def __init__(self, experiment_execution_paths):
        """
        Initializes an empty dictionary to store event execution datetimes.
        """
        self.events = defaultdict(dict)

        self.experiment_version_dir_path = \
            experiment_execution_paths.experiment_version_dir_path

    def add_event(self, event_name, datetime, subevent_name=None):
        """Adds an event and its corresponding datetime to the dictionary.

        Args:
            event_name (str): The name of the event to add.
            datetime (str): The datetime of the event, which will
                be stored as a string.
        """
        if not subevent_name:
            self.events[event_name] = datetime
        else:
            self.events[event_name][subevent_name] = datetime

    def save(self):
        """Saves the event datetimes dictionary to a JSON file.

        The file is saved to a directory defined by `CURRENT_PROTOCOL_DIR_PATH`
        with the filename `execution_datetimes.json`. The file includes an
        indentation of 4 spaces for readability.
        """
        execution_datetimes_file_path = \
            f"{self.experiment_version_dir_path}/execution_datetimes.json"
        with open(execution_datetimes_file_path, 'w') as file:
            json.dump(self.events, file, indent=4)
