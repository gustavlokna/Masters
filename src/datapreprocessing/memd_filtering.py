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
    segment, memd_params, idx, total, keep_imfs = args
    segment_T = segment.T  # (channels, samples)
    imfs = memd(segment_T,
                memd_params["num_directions"],
                memd_params["stop_criteria"],
                memd_params["stop_args"])  # (n_imfs, channels, samples)

    if imfs.shape[0] < keep_imfs:
        raise ValueError(f"[{idx+1}/{total}] Segment has only {imfs.shape[0]} IMFs, but {keep_imfs} requested")

    imfs = imfs[:keep_imfs]
    #print(f"[{idx+1}/{total}] IMFs shape after slicing: {imfs.shape}", flush=True)
    return imfs.transpose(0, 2, 1)  # → (n_imfs, samples, channels)


"""
def apply_memd_filter(X, memd_params):
    keep_imfs = memd_params["keep_imfs"]
    with Pool(processes=os.cpu_count()) as pool:
        args = [(segment, memd_params, i, len(X), keep_imfs) for i, segment in enumerate(X)]
        results = pool.map(memd_filter_segment_args, args)
    return np.stack(results)  # shape: (n_segments, n_imfs, 640, 20)
"""
def apply_memd_filter(X, memd_params):
    all_imfs = []
    keep_imfs = memd_params["keep_imfs"]

    for i, segment in enumerate(X):
        #print(f"Filtering segment {i+1}/{len(X)}")
        imfs = memd_filter_segment_args((segment, memd_params, i, len(X), keep_imfs))
        all_imfs.append(imfs)

    return np.stack(all_imfs)  # (n_segments, n_imfs, 512, 20)


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
