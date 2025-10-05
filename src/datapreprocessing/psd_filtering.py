import numpy as np
from scipy.signal import welch

def load_data(npz_path):
    """Load preprocessed data from .npz file."""
    data = np.load(npz_path)
    X, y, subject, sex = data["X"], data["y"], data["subject"], data["sex"]
    print(f"Loaded: X={X.shape}, y={y.shape}, subject={subject.shape}, sex ={sex.shape}")
    return X, y, subject, sex

def save_psd_data(output_path, X_psd, y, subject, sex, band_names):
    """Save PSD features along with labels, subjects, sex, and band names."""
    np.savez(
        output_path,
        X=X_psd,
        y=y,
        subject=subject,
        sex=sex,
        bands=band_names
    )

def compute_psd_features(X, bands, fs=256):
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
    features : array, shape (epochs, bands * channels)
        PSD band powers reshaped for classifier input.
    band_names : list of str
        Names of bands.
    """
    # transpose to (epochs, channels, samples) for Welch
    X = np.transpose(X, (0, 2, 1))

    n_epochs, n_channels, n_samples = X.shape
    band_names = list(bands.keys())
    n_bands = len(band_names)

    psd_features = np.zeros((n_epochs, n_bands, n_channels), dtype=np.float64)

    for i in range(n_epochs):
        for ch in range(n_channels):
            signal = X[i, ch, :]
            freqs, psd = welch(signal, fs=fs, nperseg=min(n_samples, fs))
            for b_idx, (fmin, fmax) in enumerate(bands.values()):
                band_filter = (freqs >= fmin) & (freqs < fmax)
                psd_features[i, b_idx, ch] = np.mean(psd[band_filter]) if band_filter.any() else 0.0

    # reshape to (epochs, bands * channels)
    features = psd_features.reshape(n_epochs, n_bands * n_channels)
    return features, band_names

def apply_psd_pipeline(config: dict) -> None:
    """Run full PSD pipeline: load, compute, save."""
    input_path = config["data"]["preprocessed"]  # use preprocessed data
    output_path = config["data"]["psd"]
    fs_config = config["data"]["fs"]
    bands = config["psd_bands"]

    X, y, subject, sex = load_data(input_path)
    X_psd, band_names = compute_psd_features(X, bands, fs=fs_config)
    save_psd_data(output_path, X_psd, y, subject, sex, band_names)