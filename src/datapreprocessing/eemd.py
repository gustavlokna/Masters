import numpy as np
import os
from multiprocessing import Pool
from utilities.eemd import eemd


def load_data(npz_path):
    data = np.load(npz_path)
    return data["X"], data["y"], data["subject"], data["sex"], data["age"]

def save_filtered_data(output_path, X_filtered, y, subject, sex, age):
    np.savez(output_path, X=X_filtered, y=y, subject=subject, sex=sex, age=age)

def eemd_filter_segment_args(args):
    segment, num_ensembles, noise_strength, max_siftings = args
    segment_T = segment.T  # (channels, samples)
    imfs = eemd(segment_T, num_ensembles, noise_strength, max_siftings)  # (n_imfs, channels, samples)
    imfs = imfs[:6]  # first 6 IMFs
    imfs_concat = np.concatenate(imfs, axis=1)  # concat along time
    return imfs_concat.T  # (samples_concat, channels)

def apply_eemd_filter(X, num_ensembles, noise_strength, max_siftings):
    all_filtered = []
    for segment in X:
        filtered = eemd_filter_segment_args((segment, num_ensembles, noise_strength, max_siftings))
        all_filtered.append(filtered)
    X_filtered = np.stack(all_filtered)
    print(f"Final filtered shape: {X_filtered.shape}")
    return X_filtered

def apply_eemd_single_band_pipeline(config: dict, subject_id: str) -> None:
    input_path = config["data"]["preprocessed_no_bp"]
    base_output_dir = config["data"]["eemd_single_band"]

    # EEMD parameters
    num_ensembles = 100
    noise_strength = 0.2
    max_siftings = 50

    # Load data
    X, y, subject, sex, age = load_data(input_path)

    # Select subject
    mask = subject == subject_id
    X_subj, y_subj, sex_subj, age_subj = X[mask], y[mask], sex[mask], age[mask]

    # Apply EEMD
    X_filtered = apply_eemd_filter(X_subj, num_ensembles, noise_strength, max_siftings)

    # Save output
    output_file = os.path.join(base_output_dir, f"eemd_filtered_{subject_id}_2.5_sepoch_256hz.npz")
    save_filtered_data(output_file, X_filtered, y_subj, subject_id, sex_subj, age_subj)
