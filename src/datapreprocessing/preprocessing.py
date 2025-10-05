import pandas as pd
import numpy as np
import os
import mne
from scipy.stats import kurtosis, zscore
from scipy.signal import welch, hamming


def detect_bad_channels(data, sfreq):
    threshold_prob = 5
    threshold_kurt = 5
    threshold_spectra = 2
    band = (20, 75)

    # Probabilistic method
    prob_scores = np.abs(zscore(data, axis=1)).mean(axis=1)
    bad_prob = np.where(prob_scores > threshold_prob)[0]

    # Kurtosis method
    kurt_scores = kurtosis(data, axis=1, fisher=False)
    bad_kurt = np.where(kurt_scores > threshold_kurt)[0]

    # Spectral method (Welch)
    f, Pxx = welch(data, fs=sfreq, axis=1)
    band_power = np.mean(Pxx[:, (f >= band[0]) & (f <= band[1])], axis=1)
    spec_z = zscore(band_power)
    bad_spec = np.where(np.abs(spec_z) > threshold_spectra)[0]

    # Final decision: use only probabilistic for removal
    return list(bad_prob)


def preprocessing(config: dict) -> None:
    print("starting preprocess")
    fs_target = config["data"]["fs"]
    crop_tmax = 119.998
    epoch_length = config["data"]["epoch_length"]
    segment_len = int(epoch_length * fs_target)

    all_X, all_y, all_subjects, all_sexes = [], [], [], []

    records_path = os.path.join(
        config["data"]["raw"],
        "Tononi Serial Awakenings-Part1-No_PSGs",
        "Tononi Serial Awakenings",
        "Records.csv"
    )
    records_df = pd.read_csv(records_path)

    for i in range(1, 37):
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
            raw.crop(tmin=0, tmax=crop_tmax)
            #raw.filter(None, 500)
            raw.resample(fs_target)

            raw.set_eeg_reference('average', projection=False)
            # TODO CHANGE WITH SIFT LATER
            raw._data = mne.filter.detrend(raw._data, order=1, axis=1)

            bad_channels = detect_bad_channels(raw._data, fs_target)
            raw.info['bads'] = [raw.ch_names[i] for i in bad_channels]
            raw.pick_types(eeg=True, exclude='bads')

            data = raw.get_data().T
            labels = raw.ch_names
            df = pd.DataFrame(data, columns=labels)

            keep_cols = [f"Chan {i}" for i in range(1, 257)]
            if set(keep_cols).issubset(df.columns):
                df = df[keep_cols]
            else:
                alt_cols = [str(i) for i in range(1, 257)]
                df = df[alt_cols]

            # Segment into epochs
            num_segments = df.shape[0] // segment_len
            clean_segments = []

            for i in range(num_segments - 1):
                seg1 = df.iloc[i * segment_len:(i + 1) * segment_len].values
                seg2 = df.iloc[(i + 1) * segment_len:(i + 2) * segment_len].values

                hamming_window = hamming(segment_len).reshape(-1, 1)
                amp1 = np.max(np.abs(seg1 * hamming_window))
                amp2 = np.max(np.abs(seg2 * hamming_window))

                if amp1 < 10 and amp2 < 10:
                    clean_segments.append(seg1 - np.mean(seg1, axis=0))
                    clean_segments.append(seg2 - np.mean(seg2, axis=0))

            if not clean_segments:
                continue

            X = np.stack(clean_segments)

            meta = records_df[records_df["Filename"] == fname]
            if meta.empty:
                raise RuntimeError(f"No metadata found for file: {fname}")

            label = meta.iloc[0]["Experience"]
            sex = meta.iloc[0]["Subject sex"]

            y = np.full((X.shape[0],), label)
            sex_array = np.full((X.shape[0],), sex)
            subject_array = np.full((X.shape[0],), subject_id)

            all_X.append(X)
            all_y.append(y)
            all_subjects.append(subject_array)
            all_sexes.append(sex_array)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    subject_all = np.concatenate(all_subjects, axis=0)
    sex_all = np.concatenate(all_sexes, axis=0)

    outpath = os.path.join(
        config["data"]["processed"],
        f"New_segmented_{epoch_length}s_epoch_{fs_target}hz.npz"
    )

    np.savez(
        outpath,
        X=X_all,
        y=y_all,
        subject=subject_all,
        sex=sex_all
    )

    print(f"Saved combined data: {outpath} with shape {X_all.shape}")
