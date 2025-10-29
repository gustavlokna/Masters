import argparse
from utilities.config import read_config
from train.csp_single_subject import test_csp_models_subject

def main(args):
    config = read_config()
    test_csp_models_subject(config, args.subject_id)

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--subject_id", required=True)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
