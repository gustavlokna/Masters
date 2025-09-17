import pandas as pd
import numpy as np
import os
import mne


def preprocessing(config: dict) -> None:
    print("starting preprocess")
    fs_target = 256
    crop_tmax = 119.998 # As time is zero indexed this corps the 120 first seconds
    segment_len = int(2 * fs_target)  # 512 samples

    all_X = []
    all_y = []
    all_subjects = []

    # Load metadata
    records_path = os.path.join(
        config["data"]["raw"],
        "Tononi Serial Awakenings-Part1-No_PSGs",
        "Tononi Serial Awakenings",
        "Records.csv"
    )
    records_df = pd.read_csv(records_path)

    # Channel selection
    top_20_idxs = config["channels"]["top_20"]
    top_20_names = [f"Chan {i}" for i in top_20_idxs]

    # Folder with .edf files
    for i in range(1, 37): # for whom ever takes over code. This is not best practice but simple (i cannot be bothered to dynamically run through subset of files)
        subject_id = f"s{str(i).zfill(2)}"
        edf_dir = os.path.join(
            config["data"]["raw"],
            f"Tononi Serial Awakenings-Part{i+1}-{subject_id}_PSGs",
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
            #df = df[top_20_names]  # keep selected channels only
            df = pd.DataFrame(data, columns=labels)

            # Try "Chan X"
            if set(top_20_names).issubset(df.columns):
                df = df[top_20_names]
            else:
                print("Trying differnt subset of names")
                fallback_names = [str(i) for i in top_20_idxs]
                if set(fallback_names).issubset(df.columns):
                    df = df[fallback_names]
                else:
                    raise RuntimeError(f"❌ Channels not found in {fname}. Tried both 'Chan X' and 'X' formats.")



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

            subject_array = np.full((num_segments,), subject_id)
            all_subjects.append(subject_array)
            subject_all = np.concatenate(all_subjects, axis=0)


    # Save data 
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    outpath = os.path.join(config["data"]["processed"], f"data_segments_combined.npz")

    np.savez(
    outpath,
    X=X_all,
    y=y_all,
    subject=subject_all  
    )

    
    print(f"Saved combined data: {outpath} with shape {X_all.shape}")

    with np.load(outpath) as data:
        print("Loaded .npz shape:", data["X"].shape, data["y"].shape)


