import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


def load_csp_data(npz_path):
    """Load CSP features + labels from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    X, y, subject, sex = data["X"], data["y"], data["subject"], data["sex"]
    print(f"Loaded CSP data: X={X.shape}, y={y.shape}")
    return X, y, subject, sex


def prepare_data(X, y, label_map):
    """Filter and remap labels according to label_map."""
    mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
    X_filtered = X[mask]
    y_filtered = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)
    return X_filtered, y_filtered


def train_and_evaluate(X, y, test_size=0.3, random_state=42):
    """Split data, train KNN, and print classification report."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=15)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, zero_division=0))

    return model


def run_training_pipeline(config):
    """Main entry point for CSP + KNN training pipeline."""
    path = config['data']['csp']
    X, y, subject, sex = load_csp_data(path)

    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Running label map: {map_name} ===")
        X_filtered, y_filtered = prepare_data(X, y, label_map)

        if len(np.unique(y_filtered)) < 2:
            print(f"Skipping {map_name} (not enough classes).")
            continue

        _ = train_and_evaluate(X_filtered, y_filtered)
