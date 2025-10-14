import numpy as np
from scipy.signal import welch


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
        age=age,
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
    features : array, shape (epochs, channels, bands)
        PSD band powers.
    band_names : list of str
        Names of bands.
    """
    X = np.transpose(X, (0, 2, 1))  # (epochs, channels, samples)
    n_epochs, n_channels, n_samples = X.shape
    band_names = list(bands.keys())
    n_bands = len(band_names)

    features = np.zeros((n_epochs, n_channels, n_bands), dtype=np.float64)

    for i in range(n_epochs):
        for ch in range(n_channels):
            signal = X[i, ch, :]
            freqs, psd = welch(signal, fs=fs, nperseg=min(n_samples, fs))
            for b_idx, (fmin, fmax) in enumerate(bands.values()):
                mask = (freqs >= fmin) & (freqs < fmax)
                features[i, ch, b_idx] = np.mean(psd[mask]) if mask.any() else 0.0

    return features, band_names


def apply_psd_pipeline(config: dict) -> None:
    """Run full PSD pipeline: load, compute, save."""
    input_path = config["data"]["preprocessed"]
    output_path = config["data"]["psd"]
    fs = config["data"]["fs"]
    bands = config["psd_bands"]

    X, y, subject, sex, age = load_data(input_path)
    X_psd, band_names = compute_psd_features(X, bands, fs)
    save_psd_data(output_path, X_psd, y, subject, sex, age, band_names)
