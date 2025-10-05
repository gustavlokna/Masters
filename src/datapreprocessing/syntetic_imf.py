import numpy as np
import random


def load_memd_data(npz_path):
    data = np.load(npz_path)
    return data["X"], data["y"], data["subject"], data["sex"]


def save_mixed_data(output_path, X, y, subject, sex, synthetic_flag):
    np.savez(
        output_path,
        X=X,
        y=y,
        subject=subject,
        sex=sex,
        synthetic=synthetic_flag,
    )
    print(f"Saved mixed data to {output_path} with shape {X.shape}")


def mix_imfs_channel_consistent(X, y, subject, sex, n_imfs=4, n_new=1000, filter_on_sex=False):
    """
    X: (n_segments, n_imfs, samples, channels)
    y: (n_segments,)
    subject: (n_segments,)
    sex: (n_segments,)
    n_imfs: number of IMFs to mix
    n_new: number of synthetic samples
    filter_on_sex: enforce same sex across IMFs
    """
    new_X, new_y, new_subject, new_sex, synthetic_flag = [], [], [], [], []

    n_segments, max_imfs, samples, channels = X.shape
    assert n_imfs <= max_imfs, "n_imfs exceeds available IMFs"

    # group by label (and sex if filtering)
    if filter_on_sex:
        groups = {}
        for i in range(n_segments):
            key = (y[i], sex[i])
            groups.setdefault(key, []).append(i)
    else:
        groups = {}
        for i in range(n_segments):
            key = y[i]
            groups.setdefault(key, []).append(i)

    for _ in range(n_new):
        # pick one label (and sex) group
        key = random.choice(list(groups.keys()))
        candidates = groups[key]

        # select different segments for each IMF index
        chosen_idx = random.sample(candidates, n_imfs)

        # collect selected IMFs (still keeping IMF axis for now)
        mixed = []
        for j, idx in enumerate(chosen_idx):
            mixed.append(X[idx, j, :, :])  # IMF j (samples, channels)

        mixed = np.stack(mixed, axis=0)  # (n_imfs, samples, channels)

        new_X.append(mixed[np.newaxis, :, :, :])
        new_y.append(y[chosen_idx[0]])
        if filter_on_sex:
            new_sex.append(sex[chosen_idx[0]])
        else:
            vals, counts = np.unique(sex[chosen_idx], return_counts=True)
            new_sex.append(vals[np.argmax(counts)])
        new_subject.append(-1)
        synthetic_flag.append(1)

    return (
        np.concatenate(new_X, axis=0),  # (n_new, n_imfs, samples, channels)
        np.array(new_y),
        np.array(new_subject),
        np.array(new_sex),
        np.array(synthetic_flag),
    )


def imf_mixing_pipeline(config: dict):
    input_path = config["data"]["memd"]
    output_path = config["data"]["mixed"]

    n_imfs = config["mixing"]["n_imfs"]
    n_new = config["mixing"]["n_new"]
    filter_on_sex = config["mixing"]["filter_on_sex"]

    X, y, subject, sex = load_memd_data(input_path)
    X_new, y_new, subject_new, sex_new, synthetic_flag = mix_imfs_channel_consistent(
        X, y, subject, sex, n_imfs=n_imfs, n_new=n_new, filter_on_sex=filter_on_sex
    )

    # --- Final reduction step: sum over first n_imfs ---
    X_orig_reduced = np.sum(X[:, :n_imfs, :, :], axis=1)  # (n_segments, samples, channels)
    X_new_reduced = np.sum(X_new[:, :n_imfs, :, :], axis=1)  # (n_new, samples, channels)

    # combine originals + synthetic
    X_all = np.concatenate([X_orig_reduced, X_new_reduced], axis=0)
    y_all = np.concatenate([y, y_new], axis=0)
    subject_all = np.concatenate([subject, subject_new], axis=0)
    sex_all = np.concatenate([sex, sex_new], axis=0)
    synthetic_all = np.concatenate([np.zeros(len(y), dtype=int), synthetic_flag], axis=0)

    save_mixed_data(output_path, X_all, y_all, subject_all, sex_all, synthetic_all)
    print(
        f"Mixed data shape: {X_all.shape}, original samples: {len(y)}, synthetic samples: {len(y_new)}"
    )