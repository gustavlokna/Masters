import argparse
from typing import Any
from utilities.config import read_config
from datapreprocessing.preprocessing import preprocessing
from datapreprocessing.memd_filtering import apply_memd_pipeline
from datapreprocessing.psd_filtering import apply_psd_pipeline
from datapreprocessing.oldTrash import test_memd_on_segment

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
    elif args.memd:
        print("running memd filtering")

        apply_memd_pipeline(config)
        
        print("running memd")
    elif args.psd:
        print("running psd feature extraction")
        apply_psd_pipeline(config)
    
    elif args.dev: 
        imfs = test_memd_on_segment("Data/processed/data_segments_combined.npz", num_directions=64)
        print("IMFs shape:", imfs.shape)
        
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
        "--memd",
        action="store_true",
        help="Enable memd filtering on preprocessed data",
    )
    parser.add_argument(
        "--psd",
        action="store_true",
        help="Enable psd feature extraction on memd filtered data",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development/testing mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)