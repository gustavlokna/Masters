import numpy as np
import os
from PyEMD import EEMD


def load_data(npz_path):
    data = np.load(npz_path)
    return data["X"], data["y"], data["subject"], data["sex"], data["age"]


def save_filtered_data(output_path, X_filtered, y, subject, sex, age):
    np.savez(output_path, X=X_filtered, y=y, subject=subject, sex=sex, age=age)


def eemd_filter_segment(segment):
    segment_T = segment.T  # (channels, samples)
    eemd = EEMD()
    eemd.trials = 50
    eemd.max_imf = 5
    imfs = eemd.eemd(segment_T)  # returns list of IMFs per channel
    imfs = imfs[:5]  # first 5 IMFs
    imfs_concat = np.concatenate(imfs, axis=1)  # concatenate along time
    return imfs_concat.T  # (samples_concat, channels)


def apply_eemd_filter(X):
    all_filtered = []
    for segment in X:
        filtered = eemd_filter_segment(segment)
        all_filtered.append(filtered)
    X_filtered = np.stack(all_filtered)
    print(f"Final filtered shape: {X_filtered.shape}")
    return X_filtered


def apply_eemd_single_band_pipeline(config: dict, subject_id: str) -> None:
    input_path = config["data"]["preprocessed_no_bp"]
    base_output_dir = "Data/eemd"
    os.makedirs(base_output_dir, exist_ok=True)

    # Load data
    X, y, subject, sex, age = load_data(input_path)

    # Select subject
    mask = subject == subject_id
    X_subj, y_subj, sex_subj, age_subj = X[mask], y[mask], sex[mask], age[mask]

    # Apply EEMD
    X_filtered = apply_eemd_filter(X_subj)

    # Save output
    output_file = os.path.join(base_output_dir, f"eemd_filtered_{subject_id}_2.5_sepoch_256hz_no_BP.npz")
    save_filtered_data(output_file, X_filtered, y_subj, subject_id, sex_subj, age_subj)
