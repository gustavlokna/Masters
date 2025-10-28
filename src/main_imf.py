import argparse
import numpy as np
from utilities.config import read_config
from datapreprocessing.imf_permutations import load_all_memd_data
from utilities.imf_permuatations import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imfs", nargs="+", type=int, required=True)
    args = parser.parse_args()
    imf_idx = args.imfs

    config = read_config()
    X, y, subject, sex, age = load_all_memd_data(config["data"]["memd_single_band"])

    X = X[:, imf_idx, :, :]
    X = np.sum(X, axis=1)

    selected = np.array(config["channels"]["top_64"]) - 1
    X = X[:, :, selected]

    X = np.transpose(X, (0, 2, 1))
    X = X[..., np.newaxis]

    train_model(config, X, y, subject, sex, age, imf_idx)

if __name__ == "__main__":
    main()
