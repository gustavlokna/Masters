### COde graveyard of functions i dont want to delete untill completly done with data preprocessing (with clear from marta)

import pandas as pd
import numpy as np
import os
import mne

def preprocessing(config: dict) -> None:
    edf_path = os.path.join(
        config["data"]["raw"],
        "Tononi Serial Awakenings-Part2-s01_PSGs",
        "Tononi Serial Awakenings",
        "Data",
        "PSG",
        "s01_ep06.edf"
    )

    raw = mne.io.read_raw_edf(edf_path, preload=True)
    print(raw.ch_names)


    data = raw.get_data().T  # shape: (samples, channels)
    # --- Check duration and number of channels ---
    # --- Check duration and number of channels ---
    n_channels = len(raw.ch_names)
    duration = raw.n_times / 500 #raw.info['sfreq']
    print(f"Channels: {n_channels}")
    print(f"Samples: {raw.n_times}, Duration: {duration:.8f} seconds")

    if duration < 120:
        print("⚠️ WARNING: Recording is shorter than 120 seconds. Skipping...")
    labels = raw.ch_names
    fs = 500#raw.info['sfreq']

    timestamps = np.arange(data.shape[0]) / fs

    df = pd.DataFrame(data, columns=labels)
    df.insert(0, "timestamp", timestamps)

    output_path = os.path.join(config["data"]["processed"], "s01_ep06_timeseries.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    raw.crop(tmin=0, tmax=119.998)  # why? TODO ask Marta why he cut out data?
    raw.filter(0.5, 35)
    raw.resample(256)
    labels = raw.ch_names
    fs = raw.info['sfreq']

    # Time column
    timestamps = np.arange(data.shape[0]) / fs

    # Build dataframe
    df = pd.DataFrame(data, columns=labels)
    df.insert(0, "timestamp", timestamps)

    # Add metadata from records
    records_path = os.path.join(
        config["data"]["raw"],
        "Tononi Serial Awakenings-Part1-No_PSGs",
        "Tononi Serial Awakenings",
        "Records.csv"
    )
    records_df = pd.read_csv(records_path)

    # Example: find metadata for 's01_ep01.edf'
    meta = records_df[records_df["Filename"] == "s01_ep06.edf"].iloc[0]
    df["subject_id"] = meta["Subject ID"]
    df["experience"] = meta["Experience"]
    df["age"] = meta["Subject age"]
    df["sex"] = meta["Subject sex"]

    # Keep only top_20 channels + timestamp + metadata
    top_20_idxs = config["channels"]["top_20"]
    top_20_names = [f"Chan {i}" for i in top_20_idxs]
    keep_cols = ["timestamp"] + top_20_names + ["subject_id", "experience", "age", "sex"]
    #df = df[keep_cols]

    # Save
    output_path = os.path.join(config["data"]["processed"], "s01_ep02_timeseries.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    

    import pandas as pd
import numpy as np
import os
import mne


def preprocessing(config: dict) -> None:
    print("starting preprocess")
    fs_target = 256
    crop_tmax = 119.998 # As time is zero indexed this corps the 120 first seconds
    epoch = 2 #TODO set as 2.5 seconds
    segment_len = int(epoch * fs_target)  # 512 samples

    all_X = []
    all_y = []

    # Load metadata
    records_path = os.path.join(
        config["data"]["raw"],
        "Tononi Serial Awakenings-Part-No_PSGs",
        "Tononi Serial Awakenings",
        "Records.csv"
    )
    records_df = pd.read_csv(records_path)

    # Channel selection
    top_20_idxs = config["channels"]["top_20"]
    top_20_names = [f"Chan {i}" for i in top_20_idxs]

    # Folder with .edf files
    subject_id = "s23"  # TODO make dynamic
    edf_dir = os.path.join(
        config["data"]["raw"],
        f"Tononi Serial Awakenings-Part24-{subject_id}_PSGs",
        "Tononi Serial Awakenings",
        "Data",
        "PSG"
    )

    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]

    for fname in edf_files:
        edf_path = os.path.join(edf_dir, fname)
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        # Breaking Adins piline structe to simplify code
        # TODO ask martha if it is okay. 
        # Re-sizeing
        raw.crop(tmin=0, tmax=crop_tmax)
        # Bandpass filtering
        raw.filter(0.5, 35)
        # Downsampeling
        raw.resample(fs_target)


        data = raw.get_data().T  # (samples, channels)
        labels = raw.ch_names

        # Keeping only Channels that Adins Greedy search algorthim idetified as most usefull
        # TODO are these still the best channels when splitting into man and woman files? 
        df = pd.DataFrame(data, columns=labels)
        df = df[top_20_names]  # keep selected channels only

        # Segment continuous EEG data into non-overlapping 2-second windows
        # Each segment contains 512 samples (2s × 256Hz), across 20 selected channels
        # Final shape: (num_segments, 512, 20) suitable for 3D ML input (e.g., CNN/RNN)
        # Segment into 2s (512 samples)
        num_segments = df.shape[0] // segment_len
        print(f"number of segments {num_segments}")

        X = np.stack([
            df.values[i * segment_len:(i + 1) * segment_len]
            for i in range(num_segments)
        ])

        # Get experience label
        meta = records_df[records_df["Filename"] == fname]
        if meta.empty:
            print("ERROR") # TODO some error catching ? 
            continue  # skip file if no metadata
        label = meta.iloc[0]["Experience"]
        y = np.full((num_segments,), label)

        # Append data to list 
        all_X.append(X)
        all_y.append(y)


    # Save data 
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    outpath = os.path.join(config["data"]["processed"], f"{subject_id}_all_segments.npz")
    np.savez(outpath, X=X_all, y=y_all)

    
    print(f"Saved combined data: {outpath} with shape {X_all.shape}")

    with np.load(outpath) as data:
        print("Loaded .npz shape:", data["X"].shape, data["y"].shape)



def test_memd_on_segment(npz_path, segment_index=1752, num_directions=16, stop_criteria="stop", stop_args=[0.075, 0.75, 0.075]):
    from utilities.memd import memd  # import here to avoid errors if unused
    import numpy as np

    # Load data
    data = np.load(npz_path)
    X = data["X"]
    print(f"Loaded data shape: {X.shape}")
    
    segment = X[segment_index].T  # (20, 512)

    # Run MEMD
    imfs = memd(segment, num_directions, stop_criteria, stop_args)
    print(f"Segment {segment_index} → IMFs shape: {imfs.shape}")  # (n_imfs, 20, 512)
    
    return imfs
