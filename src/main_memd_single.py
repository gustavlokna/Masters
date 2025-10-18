import argparse
from typing import Any
from utilities.config import read_config
from datapreprocessing.single_subject_memd_filtering import apply_memd_single_band_pipeline

def main(args: argparse.Namespace) -> None:
    """
    Main function to run MEMD filtering for a single subject in parralell on idun to concatente later.
    """
    config: dict[str, Any] = read_config()
    print(f"Running MEMD filtering for subject {args.subject_id}")
    apply_memd_single_band_pipeline(config, args.subject_id)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MEMD filtering for one subject")
    parser.add_argument("--subject_id", required=True, help="Subject ID to filter")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
