from argparse import ArgumentParser

def main():
    """
    Main function to parse command-line arguments and execute the program.

    This function initializes the argument parser, parses the command-line
    arguments, and stores them in the `args` variable.
    """
    parser = args_parse()
    args = parser.parse_args()
    # TODO: Implement the training logic
    # TODO: Implement the validation and testing logic
    # TODO: Store the model
    # TODO: Use the pretrained model to make predictions or respective evaluations
    # TODO: Use the parsed arguments to run the program
    # TODO: Set visualization options (Lightning HOWTO: https://lightning.ai/docs/pytorch/stable/visualize/logging_basic.html)
    # TODO: Set debug options (Lightning HOWTO: https://lightning.ai/docs/pytorch/stable/debug/debugging_basic.html)
    # TODO: Set verbosity (Lightning HOWTO: https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html)
    # TODO: Option for predictions and evalutions based with pretrained model on a given dataset

    # TODO: Set the code to run on any machine (including cloud) with GPU support
    # TODO: Optimizing training time
    # TODO: SLURM: https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html

    # TODO: May separate the training and testing calls or do it back-to-back

def args_parse()->ArgumentParser:
    parser = ArgumentParser()
    # Add arguments for training
    args_train(parser)
    # Add arguments for testing
    args_test(parser)
    return parser

# Function to add arguments for training
# TODO: Implement the arguments for training
def args_train(parser):
    # Trainer arguments
    #parser.add_argument("--devices", type=int, default=2)
    # Hyperparameters for the model
    #parser.add_argument("--layer_1_dim", type=int, default=128)

    return parser #May not be necessary if void function is decided to be used

# Function to add arguments for testing
# TODO: Implement the arguments for testing
def args_test(parser):
    pass

if __name__ == '__main__':
    main()
    