import numpy as np
import os
from utilities.memd import memd



def load_data(npz_path):
    data = np.load(npz_path)
    X, y, subject = data["X"], data["y"], data["subject"]
    print(f"Loaded: X={X.shape}, y={y.shape}, subject={subject.shape}")
    return X, y, subject


def memd_filter_segment(segment, memd_params):
    segment_T = segment.T  # (20, 512)
    imfs = memd(segment_T,
                memd_params["num_directions"],
                memd_params["stop_criteria"],
                memd_params["stop_args"])
    return imfs.transpose(0, 2, 1)  # shape: (n_imfs, 512, 20)


def apply_memd_filter(X, memd_params):
    all_imfs = []
    for i, segment in enumerate(X):
        print(f"Filtering segment {i}/{len(X)}")
        imfs = memd_filter_segment(segment, memd_params)
        all_imfs.append(imfs)
    return np.stack(all_imfs)  # shape: (n_segments, n_imfs, 512, 20)


def save_filtered_data(output_path, X_imfs, y, subject):
    np.savez(output_path, X=X_imfs, y=y, subject=subject)
    print(f"Saved all IMFs to {output_path}, shape: {X_imfs.shape}")


def apply_memd_pipeline(config: dict) -> None:
    input_path = config["data"]["preprocessed"]
    output_path = config["data"]["memd"]
    memd_params = config["memd_params"]

    X, y, subject = load_data(input_path)
    X_filtered = apply_memd_filter(X, memd_params)
    save_filtered_data(output_path, X_filtered, y, subject)
