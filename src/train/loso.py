import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb


def load_psd_data(npz_path):
    """Load PSD features + labels from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    X, y, subject, bands = data["X"], data["y"], data["subject"], data["bands"]
    print(f"Loaded PSD data: X={X.shape}, y={y.shape}, bands={bands}")
    return X, y, subject, bands


def prepare_data(X, y, subject, label_map):
    """Flatten features and remap labels according to label_map."""
    n_epochs, n_bands, n_channels = X.shape
    X_flat = X.reshape(n_epochs, n_bands * n_channels)

    mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
    X_filtered = X_flat[mask]
    y_filtered = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)
    subject_filtered = subject[mask]

    return X_filtered, y_filtered, subject_filtered


def leave_one_subject_out(X, y, subject_ids):
    """Train on all other subjects, test on one subject."""
    results = {}
    unique_subjects = np.unique(subject_ids)

    for subj in unique_subjects:
        X_train = X[subject_ids != subj]
        y_train = y[subject_ids != subj]
        X_test = X[subject_ids == subj]
        y_test = y[subject_ids == subj]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = xgb.XGBClassifier(
        n_estimators=1000, 
        learning_rate=0.1,
        objective='binary:logistic', 
        eval_metric='logloss',
        use_label_encoder=False, random_state=42,
        tree_method='hist', 
        device='cuda'
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        results[subj] = acc
        print(f"Subject {subj}: accuracy = {acc:.3f}")

    return results


def loso_pipeline(config):
    """Main entry point for LOSO training pipeline."""
    X, y, subject, band_names = load_psd_data(config["data"]["psd"])

    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Running label map: {map_name} ===")
        X_filtered, y_filtered, subject_filtered = prepare_data(X, y, subject, label_map)

        if len(np.unique(y_filtered)) < 2:
            print(f"Skipping {map_name} (not enough classes).")
            continue

        results = leave_one_subject_out(X_filtered, y_filtered, subject_filtered)
        print("\nFinal results per subject:")
        for subj, acc in results.items():
            print(f"Subject {subj}: {acc:.3f}")
