import numpy as np
import os
import itertools
from utilities.config import read_config

def load_all_memd_data(folder_path):
    X_list, y_list, subject_list, sex_list, age_list = [], [], [], [], []
    files = [f for f in os.listdir(folder_path) if f.endswith(".npz")]
    max_imfs = 0

    # first pass: find max number of IMFs
    for file in files:
        data = np.load(os.path.join(folder_path, file))
        X = data["X"]
        max_imfs = max(max_imfs, X.shape[1])

    # second pass: pad and collect
    for file in files:
        data = np.load(os.path.join(folder_path, file))
        X = data["X"]
        y = np.atleast_1d(data["y"])
        subject = data["subject"]
        sex = np.atleast_1d(data["sex"])
        age = np.atleast_1d(data["age"])

        # ensure subject is list-like (1D)
        if np.ndim(subject) == 0:
            subject = np.array([subject.item()] * len(y), dtype=object)

        n_imfs, samples, channels = X.shape[1:]
        if n_imfs < max_imfs:
            pad_shape = (X.shape[0], max_imfs - n_imfs, samples, channels)
            X = np.concatenate([X, np.zeros(pad_shape, dtype=X.dtype)], axis=1)
            print(f"Padded {file} from {n_imfs}→{max_imfs} IMFs")

        X_list.append(X)
        y_list.append(y)
        subject_list.append(subject)
        sex_list.append(sex)
        age_list.append(age)
        print(f"Loaded {file} with shape {X.shape}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subject = np.concatenate(subject_list, axis=0)
    sex = np.concatenate(sex_list, axis=0)
    age = np.concatenate(age_list, axis=0)
    print(f"Total combined shape: {X.shape}")
    return X, y, subject, sex, age

def create_imf_combinations(config):
    input_folder = config["data"]["memd_single_band"]
    output_folder = os.path.join("Data", "imfs_permutation")
    os.makedirs(output_folder, exist_ok=True)

    X, y, subject, sex, age = load_all_memd_data(input_folder)

    max_imfs = X.shape[1]
    n_select = min(6, max_imfs)  # first 6 imfs only

    imf_indices = list(range(n_select))
    combos = []
    for r in range(1, n_select + 1):
        combos.extend(itertools.combinations(imf_indices, r))

    print(f"Creating {len(combos)} IMF combinations")

    for c in combos:
        name = "_".join([f"imf{i}" for i in c])
        out_file = os.path.join(output_folder, f"combo_{name}.npz")
        X_new = X[:, c, :, :]  # select only chosen IMFs
        np.savez(out_file, X=X_new, y=y, subject=subject, sex=sex, age=age)
        print(out_file)

