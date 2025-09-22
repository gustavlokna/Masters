import numpy as np
import os
from multiprocessing import Pool
from utilities.memd import memd


def load_data(npz_path):
    data = np.load(npz_path)
    X, y, subject = data["X"], data["y"], data["subject"]
    print(f"Loaded: X={X.shape}, y={y.shape}, subject={subject.shape}")
    return X, y, subject


def memd_filter_segment_args(args):
    segment, memd_params = args
    segment_T = segment.T  # (channels, samples)
    imfs = memd(segment_T,
                memd_params["num_directions"],
                memd_params["stop_criteria"],
                memd_params["stop_args"])  # (n_imfs, channels, samples)
    return imfs.transpose(0, 2, 1)  # → (n_imfs, samples, channels)


def apply_memd_filter(X, memd_params):
    with Pool(processes=os.cpu_count()) as pool:
        args = [(segment, memd_params) for segment in X]
        results = pool.map(memd_filter_segment_args, args)
    return np.stack(results)  # shape: (n_segments, n_imfs, 512, 20)


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
