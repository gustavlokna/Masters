import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
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

    # if using subset of channels
    #selected = np.array(config["channels"]["top_64"]) - 1
    #X = X[:, :, selected]

    # reshape to (samples, channels, time, 1)
    X = np.expand_dims(X, axis=-1)
    return X, y, subject, sex, age


def prepare_data(X, y, label_map):
    mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
    X_filtered = X[mask]
    y_filtered = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)
    return X_filtered, y_filtered


def get_model(model_type, num_classes):
    if model_type == "SVC":
        return SVC(kernel="rbf", probability=True)
    elif model_type == "LogisticRegression":
        return LogisticRegression(max_iter=1000, multi_class='multinomial')
    elif model_type == "RandomForest":
        return RandomForestClassifier(n_estimators=1000, random_state=42)
    elif model_type == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    elif model_type == "XGBoost":
        return XGBClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=num_classes,
            eval_metric='mlogloss',
            use_label_encoder=False,
            tree_method='hist',
            device='cuda',
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def test_csp_models_subjectubject(config, file_path, subject_id, n_csp_components=64):
    X_raw, y_raw, subject, sex, age = load_raw_data(file_path, config)

    model_types = ["SVC", "XGBoost", "KNN"]
    all_results = []

    for map_name, label_map in config["label_maps"].items():

        #X, y = prepare_data(X_raw, y_raw, label_map)

        # mask for valid labels
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

        

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train_subj, y_train_subj, test_size=0.3,
            stratify=y_train_subj, random_state=42
        )

        X_tr_csp = X_tr.squeeze(-1).transpose(0,2,1)
        X_te_csp = X_te.squeeze(-1).transpose(0,2,1)
        X_val_csp = X_val_subj.squeeze(-1).transpose(0,2,1)

        csp = CSP(n_components=n_csp_components, log=True)
        X_tr_csp = csp.fit_transform(X_tr_csp, y_tr)
        X_te_csp = csp.transform(X_te_csp)
        X_val_csp = csp.transform(X_val_csp)

        scaler = StandardScaler()
        X_tr_csp = scaler.fit_transform(X_tr_csp)
        X_te_csp = scaler.transform(X_te_csp)
        X_val_csp = scaler.transform(X_val_csp)

        for m in model_types:
            model = get_model(m, nb_classes)
            model.fit(X_tr_csp, y_tr)

            preds = model.predict(X_te_csp)
            preds_excl = model.predict(X_val_csp)

            res = {
                "model_type": m,
                "label_map": map_name,
                "subject": subject_id,
                "internal_acc": accuracy_score(y_te, preds),
                "internal_recall": recall_score(y_te, preds, average="macro", zero_division=0),
                "internal_kappa": cohen_kappa_score(y_te, preds),
                "excluded_acc": accuracy_score(y_val_subj, preds_excl),
                "excluded_recall": recall_score(y_val_subj, preds_excl, average="macro", zero_division=0),
                "excluded_kappa": cohen_kappa_score(y_val_subj, preds_excl)
            }
            all_results.append(res)

    input_name = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = f"model_eval/csp/splitted/n_components_{n_csp_components}/{input_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"csp_eval_{subject_id}.csv")
    pd.DataFrame(all_results).to_csv(out_path, index=False)
