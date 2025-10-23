import numpy as np
from scipy.signal import welch
import os

def load_data(npz_path):
    """Load preprocessed data from .npz file."""
    data = np.load(npz_path)
    return data["X"], data["y"], data["subject"], data["sex"], data["age"]

def save_psd_data(output_path, X_psd, y, subject, sex, age, band_names):
    """Save PSD features along with labels, subjects, sex, age and band names."""
    np.savez(
        output_path,
        X=X_psd,
        y=y,
        subject=subject,
        sex=sex,
        age = age,
        bands=band_names
    )


def compute_psd_features(X, bands, fs):
    """
    Compute average PSD power per frequency band for each epoch and channel.

    Parameters
    ----------
    X : array, shape (epochs, samples, channels)
        Input EEG segments.
    bands : dict of str -> (fmin, fmax)
        Frequency bands.
    fs : int
        Sampling frequency.

    Returns
    -------
    features : array, shape (epochs, bands, channels)
        PSD band powers.
    band_names : list of str
        Names of bands.
    """
    # transpose to (epochs, channels, samples) for Welch
    X = np.transpose(X, (0, 2, 1))

    n_epochs, n_channels, n_samples = X.shape
    band_names = list(bands.keys())
    n_bands = len(band_names)

    features = np.zeros((n_epochs, n_bands, n_channels), dtype=np.float64)

    for i in range(n_epochs):
        for ch in range(n_channels):
            signal = X[i, ch, :]
            freqs, psd = welch(signal, fs=fs, nperseg=min(n_samples, fs))

            for b_idx, (fmin, fmax) in enumerate(bands.values()):
                band_filter = (freqs >= fmin) & (freqs < fmax)
                features[i, b_idx, ch] = np.mean(psd[band_filter]) if band_filter.any() else 0.0

    return features, band_names


def apply_psd_pipeline(config: dict, file_path) -> None:
    """Run full PSD pipeline: load, compute, save."""
        # name output file after input npz
    input_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = f"Data/psd/Psd_{input_name}.npz"

    fs = config["data"]["fs"]
    bands = config["psd_bands"]

    X, y, subject, sex , age= load_data(file_path)
    X_psd, band_names = compute_psd_features(X, bands,fs) # shape (n_epochs, n_bands, n_channels)
    print(f"Computed PSD features with shape {X_psd.shape}")
    save_psd_data(output_path, X_psd, y, subject, sex,age,  band_names)
