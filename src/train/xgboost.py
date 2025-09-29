import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report

def train_raw_memd_pipeline(config: dict, keep_imfs=4):
    """
    Train ML model directly on raw MEMD data.
    Keeps only the first `keep_imfs` IMFs per epoch.
    Optionally excludes specific segments by index.
    """

    # Load MEMD-filtered data
    data = np.load(config["data"]["memd"])
    exclude_segments = config["exclude_segments"] 
    X_raw, y, subject = data["X"], data["y"], data["subject"]
    # shape: (n_segments, n_imfs, samples, channels)

    # Keep only the first `keep_imfs`
    X_raw = X_raw[:, :keep_imfs, :, :]

    # Exclude unwanted segments
    if exclude_segments:
        mask = np.ones(len(X_raw), dtype=bool)
        mask[exclude_segments] = False
        X_raw, y, subject = X_raw[mask], y[mask], subject[mask]

    # Flatten each epoch → row in X
    n_samples = X_raw.shape[0]
    X = X_raw.reshape(n_samples, -1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        tree_method="hist",
        device="cuda"
    )
    model.fit(X_train, y_train)

    # Predict + eval
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))



def train_raw_pipeline(config: dict):
    """
    Train ML model directly on raw preprocessed EEG segments (no MEMD).
    Each epoch is treated as one instance (flattened across channels and samples).
    Optionally excludes specific segments by index.
    """

    # Load RAW (preprocessed, not MEMD) data
    data = np.load(config["data"]["preprocessed"])
    exclude_segments = config["exclude_segments"] 
    X_raw, y, subject = data["X"], data["y"], data["subject"]
    # shape: (n_segments, samples, channels)

    # Exclude unwanted segments
    if exclude_segments:
        mask = np.ones(len(X_raw), dtype=bool)
        mask[exclude_segments] = False
        X_raw, y, subject = X_raw[mask], y[mask], subject[mask]

    # Flatten each epoch → row in X
    n_samples = X_raw.shape[0]
    X = X_raw.reshape(n_samples, -1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        predictor="gpu_predictor",
        device="cuda"
    )

    model.fit(X_train, y_train)

    # Predict + eval
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))