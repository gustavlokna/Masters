import numpy as np
import os
from utilities.memd import memd



def load_data(npz_path):
    data = np.load(npz_path)
    X, y, subject = data["X"], data["y"], data["subject"]
    print(f"Loaded: X={X.shape}, y={y.shape}, subject={subject.shape}")
    return X, y, subject


def memd_filter_segment(segment, memd_params):
    # segment: (512, 20)
    print("hei")
    segment_T = segment.T  # (20, 512)

    imfs = memd(segment_T,
                memd_params["num_directions"],
                memd_params["stop_criteria"],
                memd_params["stop_args"])
    reconstructed = np.sum(imfs, axis=0)  # (20, 512)
    return reconstructed.T  # (512, 20)


def apply_memd_filter(X, memd_params):
    filtered_segments = []
    for i, segment in enumerate(X):
        #if i % 100 == 0:
        print(f"Filtering segment {i}/{len(X)}")
        filtered = memd_filter_segment(segment, memd_params)
        filtered_segments.append(filtered)
    return np.stack(filtered_segments)


def save_filtered_data(output_path, X_filtered, y, subject):
    np.savez(output_path, X=X_filtered, y=y, subject=subject)
    print(f"Saved filtered data to {output_path} with shape {X_filtered.shape}")


def apply_memd_pipeline(config: dict) -> None:
    input_path = config["data"]["preprocessed"]
    output_path = config["data"]["memd"]
    memd_params = config["memd_params"]

    X, y, subject = load_data(input_path)
    X_filtered = apply_memd_filter(X, memd_params)
    save_filtered_data(output_path, X_filtered, y, subject)
