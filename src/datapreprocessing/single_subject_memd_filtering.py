import numpy as np
import os
from multiprocessing import Pool
from utilities.memd import memd


def load_data(npz_path):
    ## Load preprocessed data from .npz file.
    data = np.load(npz_path); 
    return data["X"], data["y"], data["subject"], data["sex"], data["age"]

def save_filtered_data(output_path, X_filtered, y, subject, sex, age):
    np.savez(output_path, X=X_filtered, y=y, subject=subject, sex=sex, age = age)

def memd_filter_segment_args(args):
    segment, memd_params = args
    segment_T = segment.T  # (channels, samples)
    imfs = memd(segment_T,
                memd_params["num_directions"],
                memd_params["stop_criteria"],
                memd_params["stop_args"])  # (n_imfs, channels, samples)
    return imfs.transpose(0, 2, 1)  # → (n_imfs, samples, channels)

def apply_memd_filter(X, memd_params):
    all_imfs = []
    tmp_results = []
    max_imfs = 0
    min_imfs = float("inf")
    padded_segments = []

    # First pass: compute IMFs per segment
    for idx, segment in enumerate(X):
        imfs = memd_filter_segment_args((segment, memd_params))
        tmp_results.append(imfs)
        n_imfs = imfs.shape[0]
        max_imfs = max(max_imfs, n_imfs)
        min_imfs = min(min_imfs, n_imfs)

    # Second pass: pad to max_imfs
    for idx, imfs in enumerate(tmp_results):
        n_imfs, samples, channels = imfs.shape
        if n_imfs < max_imfs:
            pad_shape = (max_imfs - n_imfs, samples, channels)
            imfs = np.concatenate([imfs, np.zeros(pad_shape, dtype=imfs.dtype)], axis=0)
            padded_segments.append(idx)
        all_imfs.append(imfs)

    print(f"Final shape: {(len(all_imfs), max_imfs, samples, channels)}")
    print(f"Min number of IMFs: {min_imfs}")
    if padded_segments:
        print(f"Segments needing padding: {padded_segments}")
    else:
        print("No segments needed padding.")

    return np.stack(all_imfs)  # (n_segments, max_imfs, samples, channels)


def apply_memd_single_band_pipeline(config: dict, subject_id: str) -> None:
    input_path = config["data"]["preprocessed_no_bp"]
    base_output_dir = config["data"]["memd_single_band"]
    memd_params = config["memd_params"]

    # Load data
    X, y, subject, sex, age = load_data(input_path)

    # Filter data for the given subject
    mask = subject == subject_id
    X_subj, y_subj, sex_subj, age_subj = X[mask], y[mask], sex[mask], age[mask]

    # Apply MEMD filter
    X_filtered = apply_memd_filter(X_subj, memd_params)

    # Save file as memd_filtered_SUBJECT_ID.npz
    output_file = os.path.join(base_output_dir, f"memd_filtered_{subject_id}.npz")
    save_filtered_data(output_file, X_filtered, y_subj, subject_id, sex_subj, age_subj)

