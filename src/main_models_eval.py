import argparse
from typing import Any
import os
from utilities.config import read_config
from train.model_eval import models_eval


def main(args: argparse.Namespace) -> None:
    """
    Main function to run DeepConvNet evaluation for a single .npz file in parallel on Idun.
    """
    config: dict[str, Any] = read_config()
    root_path = config["data"]["psd"]
    file_path = os.path.join(root_path, args.file)

    print(f"Running Model evaluation evaluation for file {file_path}")
    models_eval(config, file_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepConvNet evaluation for one dataset file")
    parser.add_argument("--file", required=True, help="Filename (inside Data/processed) to evaluate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
