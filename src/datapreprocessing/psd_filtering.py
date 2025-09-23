import numpy as np
from scipy.signal import welch


def load_data(npz_path):
    data = np.load(npz_path)
    X, y, subject = data["X"], data["y"], data["subject"]
    print(f"Loaded: X={X.shape}, y={y.shape}, subject={subject.shape}")
    return X, y, subject

def save_psd_data(output_path, X_psd, y, subject):
    np.savez(output_path, X=X_psd, y=y, subject=subject)
    print(f"Saved PSD features to {output_path} with shape {X_psd.shape}")

def compute_psd_features(X, bands, fs=256):
    n_epochs, n_samples, n_channels = X.shape
    features = np.zeros((n_epochs, n_channels * len(bands)))

    # TODO UNDERSTAND THIS FUNCTION BETTER
    for i in range(n_epochs):
        for ch in range(n_channels):
            f, Pxx = welch(X[i, :, ch], fs=fs, nperseg=fs)
            band_powers = []
            for band in bands.values():
                mask = (f >= band[0]) & (f <= band[1])
                band_power = np.mean(Pxx[mask])
                band_powers.append(band_power)
            features[i, ch * len(bands):(ch + 1) * len(bands)] = band_powers
    return features



def apply_psd_pipeline(config: dict) -> None:
    input_path = config["data"]["memd"]
    output_path = config["data"]["psd"]

    bands = config["psd_bands"]  # Read from config

    # Load MEMD data: shape (n_segments, n_imfs, 640, 20)
    X, y, subject = load_data(input_path)

    # Reconstruct signal from IMF1–4: sum over axis=1 (IMFs)
    X_reconstructed = X[:, :4, :, :].sum(axis=1)  # shape: (n_segments, 640, 20)

    # Compute PSD features
    X_psd = compute_psd_features(X_reconstructed, bands)

    # Save features
    save_psd_data(output_path, X_psd, y, subject)
