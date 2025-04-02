# Experiments Configuration

This document explains how to configure experiments in this project.

## Directory Structure

The `src/config/experiments` directory contains YAML configuration files for different experiments. Each experiment is defined in its own file, following a structured format.

## Configuration Format

Each configuration file should be written in YAML format and include the following fields:

- **`protocol`**: Specifies the protocol to be used for the experiment (e.g., `ProtocolEfficientNet`).
- **`protocol_params`**: A dictionary of parameters required by the specified protocol.
    - Example:
        ```yaml
        protocol: "ProtocolEfficientNet"
        protocol_params:
          num_classes: 1
          num_channels: 1
          model_name: "efficientnet-b0"
        ```

## Adding a New Experiment

1. Create a new YAML file in the `src/config/experiments` directory.
2. Define the experiment's settings using the format described above.
3. Save the file with a descriptive name (e.g., `exp1.yml`).

## Running Experiments

To run an experiment:
1. Ensure the configuration file is correctly set up.
2. Use the `main.py` script to load the configuration and execute the experiment.

Example:
```bash
python main.py --config_file src/config/experiments/exp1.yml
```

## Notes

- Only include parameters required by the protocol in the configuration file. Including unnecessary parameters will result in errors.
- Ensure all paths and dependencies are correctly set up before running the experiment.

For further details, refer to the project's main documentation.
