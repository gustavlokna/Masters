import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def export_imf_metadata_detailed(config: dict):
    """
    Load MEMD-filtered data and export IMF counts.
    Includes total IMF count and per-channel IMF counts for top_20 channels.
    """
    memd_file = config["data"]["memd"]
    top_channels = config["channels"]["top_20"]
    

    # Load MEMD-filtered data
    data = np.load(memd_file)
    X = data["X"]           # shape: (n_segments, max_imfs, samples, channels)
    subject = data["subject"]
    y = data["y"]

    records = []
    for seg_idx, (seg_imfs, subj, label) in enumerate(zip(X, subject, y)):
        # Total IMF count across all channels
        n_imfs_total = sum(np.any(imf != 0) for imf in seg_imfs)

        # Per-channel IMF counts (restricted to top_20 channels)
        channel_counts = {}
        for local_ch, global_ch in enumerate(top_channels):
            n_imfs_ch = sum(np.any(imf[:, local_ch] != 0) for imf in seg_imfs)
            channel_counts[f"channel_{global_ch}_imfs"] = n_imfs_ch

        record = {
            "subject_id": subj,
            "segment_number": seg_idx,
            "label": label,
            "n_imfs_total": n_imfs_total,
            **channel_counts
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Paths from config
    out_csv = config["output"]["imf_metadata_csv"]
    out_excel = config["output"]["imf_metadata_excel"]

    # Ensure directories exist
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if out_excel:
        os.makedirs(os.path.dirname(out_excel), exist_ok=True)

    # Save
    df.to_csv(out_csv, index=False)
    if out_excel:
        df.to_excel(out_excel, index=False)
    return



def plot_imfs_for_segments(config: dict):
    """
    Plot original signal + IMFs for selected segments and save in folder structure:
    root/subject/segment/channel/IMFs.png
    """
    memd_file = config["data"]["memd"]
    raw_file = config["data"]["preprocessed"]
    top_channels = config["channels"]["top_20"]
    root_out = config["output"]["imf_plot_root"]
    segment_numbers = config["output"]["imf_plot_segments"]

    # Load MEMD-filtered data
    memd_data = np.load(memd_file)
    X_memd = memd_data["X"]       # (n_segments, max_imfs, samples, channels)
    subject = memd_data["subject"]
    y = memd_data["y"]

    # Load original preprocessed data
    raw_data = np.load(raw_file)
    X_raw = raw_data["X"]         # (n_segments, samples, channels)

    for seg_idx in segment_numbers:
        seg_imfs = X_memd[seg_idx]             # (max_imfs, samples, channels)
        seg_raw = X_raw[seg_idx]               # (samples, channels)
        subj = subject[seg_idx]
        label = y[seg_idx]

        for ch_local, ch_global in enumerate(top_channels):
            channel_data = seg_imfs[:, :, ch_local]  # (max_imfs, samples)
            raw_channel = seg_raw[:, ch_local]       # (samples,)

            if not np.any(channel_data):  # skip empty
                continue

            out_dir = os.path.join(root_out, str(subj), f"segment_{seg_idx}")
            os.makedirs(out_dir, exist_ok=True)

            # Number of IMFs (exclude padded)
            valid_imfs = [imf for imf in channel_data if np.any(imf)]
            nrows = len(valid_imfs) + 1  # original + all IMFs

            # Create stacked subplot: original on top, IMFs below
            fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 2*nrows), sharex=True)

            # Original signal
            axes[0].plot(raw_channel, color="black")
            axes[0].set_title(f"Original Signal | Subject {subj} | Segment {seg_idx} | Channel {ch_global} | Label={label}")
            axes[0].set_ylabel("Amplitude")

            # IMFs
            for i, imf in enumerate(valid_imfs, start=1):
                axes[i].plot(imf)
                axes[i].set_title(f"IMF {i-1}")
                axes[i].set_ylabel("Amp")

            axes[-1].set_xlabel("Samples (2.5s epoch)")
            plt.tight_layout()

            out_path = os.path.join(out_dir, f"channel_{ch_global}_IMFs.png")
            plt.savefig(out_path)
            plt.close()

    print(f"✅ Saved IMF + original signal plots under {root_out}")
