import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb


def load_psd_data(npz_path):
    """Load PSD features + labels from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    X, y, subject, bands, sex = data["X"], data["y"], data["subject"], data["bands"], data["sex"]
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


def train_model(X_train, y_train):
    """Train XGBoost model."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

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
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate model and return accuracy."""
    X_test = scaler.transform(X_test)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(classification_report(y_test, preds, zero_division=0))
    return acc


def leave_one_subject_out_pipeline(config):
    """Train base model, then perform leave-one-subject-out evaluation."""
    X, y, subject, band_names, sex = load_psd_data(config["data"]["psd"])

    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Running label map: {map_name} ===")
        X_filtered, y_filtered = prepare_data(X, y, label_map)
        subj_filtered = np.array(subject)[[label_map.get(lbl, None) is not None for lbl in y]]

        if len(np.unique(y_filtered)) < 2:
            print(f"Skipping {map_name} (not enough classes).")
            continue

        # Step 1: Train baseline model (70/30 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.3, stratify=y_filtered, random_state=42
        )
        print("\n=== Baseline Model (70/30 Split) ===")
        model, scaler = train_model(X_train, y_train)
        base_acc = evaluate_model(model, scaler, X_test, y_test)
        print(f"Baseline Accuracy: {base_acc:.4f}")

        # Step 2: Leave-One-Subject-Out evaluation
        subjects = np.unique(subj_filtered)
        subject_results = []

        for subj in subjects:
            train_mask = subj_filtered != subj
            test_mask = subj_filtered == subj

            X_train_subj = X_filtered[train_mask]
            y_train_subj = y_filtered[train_mask]
            X_val_subj = X_filtered[test_mask]
            y_val_subj = y_filtered[test_mask]

            # Re-train model on remaining subjects (with internal 70/30 split)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train_subj, y_train_subj, test_size=0.3, stratify=y_train_subj, random_state=42
            )
            model, scaler = train_model(X_tr, y_tr)
            internal_acc = evaluate_model(model, scaler, X_te, y_te)
            excl_acc = evaluate_model(model, scaler, X_val_subj, y_val_subj)

            subject_results.append({
                "subject": subj,
                "internal_acc": internal_acc,
                "excluded_acc": excl_acc
            })

            print(f"Subject {subj}: internal_acc={internal_acc:.4f}, excluded_acc={excl_acc:.4f}")

        df_results = pd.DataFrame(subject_results)
        print("\n=== Leave-One-Subject-Out Summary ===")
        print(df_results)
        print(f"Mean internal acc: {df_results['internal_acc'].mean():.4f}")
        print(f"Mean excluded acc: {df_results['excluded_acc'].mean():.4f}")
