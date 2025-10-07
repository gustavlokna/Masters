import argparse
from typing import Any
from utilities.config import read_config
from datapreprocessing.preprocessing import preprocessing


def main(args: argparse.Namespace) -> None:
    """
    Main function for the ML Piple

    This function coordinates the execution of various stages in the classification
    pipeline, including data preprocessing, feature enhancement, model training, prediction,
    and evaluation. Each step is activated based on the provided command line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments parsed by argparse that determine which parts of the pipeline
        are activated (preprocessing, adding features, training, predicting, evaluating).

    Returns
    -------
    None

    Notes
    -----
    The function begins by reading the configuration settings. Based on the command line
    arguments, it executes one or more of the following steps:
    
    TODO


    If no valid command line arguments are provided, the function will exit with a guidance message.
    """
    config: dict[str, Any] = read_config()

    if args.preprocess:
        #_run_preprocessing(args.building, config["data"], args.include_elhub)
        preprocessing(config)
        print("completed data preprocessing")


    elif args.train: 
        print("running training")
        
    elif args.dev: 
        print("development mode")
        
    else:
        print("No valid arguments provided. Use --help for usage information.")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.

    Notes
    -----
    This function sets up the argument parser with the following options:
    --preprocess : Flag to enable preprocessing of raw EEG data.
    """
    parser = argparse.ArgumentParser(description="Sensor anomaly detection pipeline")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Enable preprocessing of sensor and HVAC data",
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development/testing mode",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable training of model on memd filtered data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)