import pandas as pd
import numpy as np
import os
import mne


def preprocessing(config: dict) -> None:
    print("starting preprocess")
    fs_target = config["data"]["fs"]
    crop_tmax = 119.998 # As time is zero indexed this corps the 120 first seconds
    epoch_length = config["data"]["epoch_length"] # seconds
    segment_len = int(epoch_length * fs_target)  # 640 samples

    all_X = []
    all_y = []
    all_subjects = []
    all_sexes = []
    all_ages = []

    # Load metadata
    records_path = os.path.join(
        config["data"]["raw"],
        "Tononi Serial Awakenings-Part1-No_PSGs",
        "Tononi Serial Awakenings",
        "Records.csv"
    )
    records_df = pd.read_csv(records_path)


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


            # keep only Chan 1–256 (or 1–256 if no 'Chan ' prefix)
            keep_cols = [f"Chan {i}" for i in range(1, 257)]
            if set(keep_cols).issubset(df.columns):
                df = df[keep_cols]
            else:
                alt_cols = [str(i) for i in range(1, 257)]
                df = df[alt_cols]


            # Segment continuous EEG data into non-overlapping 2-second windows
            # Each segment contains 640 samples (2.5s × 256Hz), across 20 selected channels
            # Final shape: (num_segments, 640, 20) suitable for 3D ML input (e.g., CNN/RNN)
            # Segment into 2.5s (640 samples)
            num_segments = df.shape[0] // segment_len
            print(f"number of segments {num_segments}")

            X = np.stack([
                df.values[i * segment_len:(i + 1) * segment_len]
                for i in range(num_segments)
            ])

            # Get metadata (experience + sex)
            meta = records_df[records_df["Filename"] == fname]
            if meta.empty:
                raise RuntimeError(f"❌ No metadata found for file: {fname}")
            label = meta.iloc[0]["Experience"]
            sex = meta.iloc[0]["Subject sex"]  # 0=female, 1=male
            age = meta.iloc[0]["Subject age"]   
            y = np.full((num_segments,), label)
            
            sex_array = np.full((num_segments,), sex)
            subject_array = np.full((num_segments,), subject_id)
            age_array = np.full((num_segments,), age)
            # Append to lists
            all_X.append(X)
            all_y.append(y)
            all_subjects.append(subject_array)
            all_sexes.append(sex_array)
            all_ages.append(age_array)

    # Concatenate all subjects
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    subject_all = np.concatenate(all_subjects, axis=0)
    sex_all = np.concatenate(all_sexes, axis=0)
    age_all = np.concatenate(all_ages, axis=0)
    # Save data
    outpath = os.path.join(
        config["data"]["processed"],
        f"segmented_{epoch_length}s_epoch_{fs_target}hz.npz"
    )

    np.savez(
        outpath,
        X=X_all,
        y=y_all,
        subject=subject_all,
        sex=sex_all,
        age=age_all
    )

    print(f"Saved combined data: {outpath} with shape {X_all.shape}")