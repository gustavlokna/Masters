import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd

sns.set(style="whitegrid")

def load_psd_file(path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["subject"], data["sex"], data["age"], data["bands"]

def band_subject_channel_correlation(X, y, subject_ids, band_names, output_dir):
    labels = np.unique(y)
    subjects = np.unique(subject_ids)
    n_bands = X.shape[1]
    n_channels = X.shape[2]

    for label in labels:
        label_mask = (y == label).astype(int)

        for b, band in enumerate(band_names):
            corr_matrix = np.zeros((len(subjects), n_channels))

            for i, subj in enumerate(subjects):
                subj_mask = subject_ids == subj
                if np.sum(subj_mask) < 3:
                    corr_matrix[i, :] = np.nan
                    continue

                for c in range(n_channels):
                    band_power = X[subj_mask, b, c]
                    label_subj = label_mask[subj_mask]
                    if len(np.unique(label_subj)) < 2:
                        corr_matrix[i, c] = np.nan
                        continue
                    r, _ = pearsonr(band_power, label_subj)
                    corr_matrix[i, c] = r

            df = pd.DataFrame(corr_matrix, index=[f"S{subj}" for subj in subjects])
            df.to_csv(os.path.join(output_dir, f"label{label}_band_{band}_subject_channel_corr.csv"))

            plt.figure(figsize=(14, 6))
            sns.heatmap(df, cmap="coolwarm", center=0, xticklabels=True, yticklabels=True)
            plt.title(f"Correlation: {band} band power vs Label {label} (per Subject × Channel)")
            plt.xlabel("Channel")
            plt.ylabel("Subject")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"label{label}_band_{band}_subject_channel_corr.png"))
            plt.close()

def run_subjectwise_band_corr(npz_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X, y, subject, _, _, bands = load_psd_file(npz_path)
    band_names = [b.decode("utf-8") if isinstance(b, bytes) else b for b in bands]
    band_subject_channel_correlation(X, y, subject, band_names, output_dir)
