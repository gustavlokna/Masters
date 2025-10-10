import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import xgboost as xgb


def load_psd_data(npz_path):
    """Load PSD features + labels from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    X, y, subject, bands, sex  = data["X"], data["y"], data["subject"], data["bands"], data["sex"]
    print(f"Loaded PSD data: X={X.shape}, y={y.shape}, bands={bands}")
    return X, y, subject, bands, sex


def prepare_data(X, y, label_map):
    """Flatten features and remap labels according to label_map."""
    n_epochs, n_bands, n_channels = X.shape
    X_flat = X.reshape(n_epochs, n_bands * n_channels)

    mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
    X_filtered = X_flat[mask]
    y_filtered = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)

    return X_filtered, y_filtered


def train_and_evaluate(X, y, band_names, test_size=0.3, random_state=42):
    """Split data, train XGBoost, and print classification report."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=1000, 
        learning_rate=0.1,
        objective='binary:logistic', 
        eval_metric='logloss',
        use_label_encoder=False, 
        random_state=42,
        tree_method='hist', 
        device='cuda'
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, zero_division=0))

    # Feature importance
    

    return model


def subject_importance(config):
    """Main entry point for training pipeline."""
    X, y, subject, band_names, sex = load_psd_data(config["data"]["psd"])

    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Running label map: {map_name} ===")
        X_filtered, y_filtered = prepare_data(X, y, label_map)

        if len(np.unique(y_filtered)) < 2:
            print(f"Skipping {map_name} (not enough classes).")
            continue

        train_and_evaluate(X_filtered, y_filtered, band_names)
