# src/main_memd_loso_deepconv.py
import argparse
import numpy as np
import os
from utilities.config import read_config
from train.deep_conv_single import test_deep_conv_subject   # your existing deep conv trainer

def load_and_sum_memd(folder_path):
    X_list, y_list, subject_list, sex_list, age_list = [], [], [], [], []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npz")])
    for f in files:
        data = np.load(os.path.join(folder_path, f), allow_pickle=True)
        X = data["X"]                  # (segments, n_imfs, samples, chans)
        y = np.atleast_1d(data["y"])
        subject = data["subject"]
        sex = np.atleast_1d(data["sex"])
        age = np.atleast_1d(data["age"])

        # sum IMFs → (segments, samples, chans)
        X_sum = np.sum(X, axis=1)
        # ensure subject array
        if np.ndim(subject) == 0:
            subject = np.array([subject.item()] * len(y), dtype=object)

        X_list.append(X_sum)
        y_list.append(y)
        subject_list.append(subject)
        sex_list.append(sex)
        age_list.append(age)
        print(f"Loaded {f} with shape {X_sum.shape}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subject = np.concatenate(subject_list, axis=0)
    sex = np.concatenate(sex_list, axis=0)
    age = np.concatenate(age_list, axis=0)
    print(f"Total combined shape: {X.shape}")
    return X, y, subject, sex, age


def main(args):
    config = read_config()
    folder_path = config["data"]["memd_single_band"]
    X, y, subject, sex, age = load_and_sum_memd(folder_path)
    test_deep_conv_subject(config, (X, y, subject, sex, age), args.subject_id)


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--subject_id", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
