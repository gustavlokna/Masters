import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mne.decoding import CSP
import os


def load_raw_data(npz_path, config):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    subject = data["subject"]
    sex = data["sex"]
    age = data["age"]

    print(f"Loaded RAW data: X={X.shape}, y={y.shape}")
    X = np.expand_dims(X, axis=-1)
    return X, y, subject, sex, age


def prepare_data(X, y, label_map):
    mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
    X_filtered = X[mask]
    y_filtered = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)
    return X_filtered, y_filtered


def test_csp_models_subject(config, file_path, subject_id, n_csp_components=256):
    X_raw, y_raw, subject, sex, age = load_raw_data(file_path, config)

    all_results = []

    for map_name, label_map in config["label_maps"].items():
        valid_mask = np.array([label_map.get(lbl, None) is not None for lbl in y_raw])

        X = X_raw[valid_mask]
        y = np.array([label_map[lbl] for lbl in y_raw[valid_mask]], dtype=int)
        subject_filt = np.array(subject)[valid_mask]
        nb_classes = len(np.unique(y))

        # LOSO masks
        train_mask = subject_filt != subject_id
        test_mask = subject_filt == subject_id

        X_train_subj = X[train_mask]
        y_train_subj = y[train_mask]
        X_val_subj = X[test_mask]
        y_val_subj = y[test_mask]

        # internal split inside training set
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train_subj, y_train_subj, test_size=0.3,
            stratify=y_train_subj, random_state=42
        )

        X_tr_csp = X_tr.squeeze(-1).transpose(0, 2, 1)
        X_te_csp = X_te.squeeze(-1).transpose(0, 2, 1)
        X_val_csp = X_val_subj.squeeze(-1).transpose(0, 2, 1)

        csp = CSP(n_components=n_csp_components, log=True)
        X_tr_csp = csp.fit_transform(X_tr_csp, y_tr)
        X_te_csp = csp.transform(X_te_csp)
        X_val_csp = csp.transform(X_val_csp)

        scaler = StandardScaler()
        X_tr_csp = scaler.fit_transform(X_tr_csp)
        X_te_csp = scaler.transform(X_te_csp)
        X_val_csp = scaler.transform(X_val_csp)

        # --- KMeans instead of supervised models ---
        kmeans = KMeans(n_clusters=nb_classes, random_state=42)
        kmeans.fit(X_tr_csp)

        # predictions
        preds = kmeans.predict(X_te_csp)
        preds_excl = kmeans.predict(X_val_csp)

        # handle label flip
        acc_te = max(
            accuracy_score(y_te, preds),
            accuracy_score(y_te, 1 - preds)
        )
        acc_val = max(
            accuracy_score(y_val_subj, preds_excl),
            accuracy_score(y_val_subj, 1 - preds_excl)
        )

        res = {
            "model_type": "KMeans",
            "label_map": map_name,
            "subject": subject_id,
            "internal_acc": acc_te,
            "internal_recall": recall_score(y_te, preds, average="macro", zero_division=0),
            "internal_kappa": cohen_kappa_score(y_te, preds),
            "excluded_acc": acc_val,
            "excluded_recall": recall_score(y_val_subj, preds_excl, average="macro", zero_division=0),
            "excluded_kappa": cohen_kappa_score(y_val_subj, preds_excl)
        }
        all_results.append(res)

    input_name = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = f"model_eval/kmeans/splitted/{input_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"kmeans_eval_{subject_id}.csv")
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"Saved KMeans LOSO results to {out_path}")
