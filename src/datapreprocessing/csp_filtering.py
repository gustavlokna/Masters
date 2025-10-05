import os
import numpy as np
from mne.decoding import CSP
import mne


def load_data(npz_path):
    """Load preprocessed data from .npz file."""
    data = np.load(npz_path)
    return data["X"], data["y"], data["subject"], data["sex"]


def bandpass_filter(X, fs, l_freq, h_freq):
    T, S, C = X.shape  # trials, samples, channels
    X_reshaped = X.transpose(0, 2, 1).reshape(-1, S)  # (T*C, S)
    X_filt = mne.filter.filter_data(X_reshaped, fs, l_freq, h_freq, verbose=False)
    return X_filt.reshape(T, C, S).transpose(0, 2, 1)  # back to (T, S, C)


def extract_csp_features(X, y, fs, config, band=None):
    if band is not None:
        l_freq, h_freq = band
        X = bandpass_filter(X, fs, l_freq, h_freq)

    X = X.transpose(0, 2, 1)  # CSP wants (trials, channels, samples)

    csp = CSP(n_components=config["csp"]["n_components"],
              reg=config["csp"].get("reg", None),
              log=True,
              cov_est=config["csp"].get("cov_est", 'concat'))

    X_csp = csp.fit_transform(X, y)
    return X_csp


def run_csp_extraction(config: dict):
    print("Running CSP extraction")

    fs = config["data"]["fs"]
    epoch_length = config["data"]["epoch_length"]
    input_path = config["data"]["preprocessed"]
    X, y, subject, sex = load_data(input_path)

    # Option A: single-band CSP (e.g., 8–30 Hz)
    print("→ Single-band CSP")
    band = config["csp"]["band"]
    X_csp = extract_csp_features(X, y, fs, config, band=band)

    save_path = os.path.join(config["data"]["processed"], f"CSP_hz_{fs}_epoch_{epoch_length}s_singleband.npz")
    np.savez(save_path, X=X_csp, y=y, subject=subject, sex=sex)
    print(f"✅ Saved single-band CSP: {save_path} shape={X_csp.shape}")

    # Option B: FBCSP
    print("→ Multi-band CSP (FBCSP)")
    X_fbcsp_all = []
    for band in config["fbcsp"]["bands"]:
        X_band = extract_csp_features(X, y, fs, config, band=band)
        X_fbcsp_all.append(X_band)
    X_fbcsp = np.concatenate(X_fbcsp_all, axis=1)

    save_path = os.path.join(config["data"]["processed"], f"CSP_hz_{fs}_epoch_{epoch_length}s_fbcsp.npz")
    np.savez(save_path, X=X_fbcsp, y=y, subject=subject, sex=sex)
    print(f"✅ Saved FBCSP: {save_path} shape={X_fbcsp.shape}")
