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
        return KNeighborsClassifier(n_neighbors=10)
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


def test_csp_models_raw(config, file_path, n_csp_components=256):
    X_raw, y_raw, subject, sex, age = load_raw_data(file_path, config)
    all_results = []
    # ["SVC", "LogisticRegression", "RandomForest", "XGBoost"]
    model_types = ["SVC", "XGBoost"]

    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Label map: {map_name} ===")
        X, y = prepare_data(X_raw, y_raw, label_map)
        subj_filtered = np.array(subject)[[label_map.get(lbl, None) is not None for lbl in y_raw]]
        subjects = np.unique(subj_filtered)
        nb_classes = len(np.unique(y))

        # reshape for CSP: (samples, channels, time)
        X_csp_input = X.squeeze(-1).transpose(0, 2, 1)  # (samples, time, channels)

        csp = CSP(n_components=n_csp_components, log=True)
        X_csp = csp.fit_transform(X_csp_input, y)

        # baseline 70/30
        X_train, X_test, y_train, y_test = train_test_split(
            X_csp, y, test_size=0.3, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for model_type in model_types:
            print(f"\n=== {model_type} (Baseline) ===")
            model = get_model(model_type, nb_classes)
            model.fit(X_train, y_train)
            base_preds = model.predict(X_test)

            base_acc = accuracy_score(y_test, base_preds)
            base_recall = recall_score(y_test, base_preds, average="macro", zero_division=0)
            base_kappa = cohen_kappa_score(y_test, base_preds)
            print(classification_report(y_test, base_preds, zero_division=0))

            subj_results = []
            """
            for subj in subjects:
                train_mask = np.array([subj not in str(s) for s in subj_filtered])
                test_mask = np.array([str(s) == str(subj) for s in subj_filtered])

                X_train_subj = X[train_mask]
                y_train_subj = y[train_mask]
                X_val_subj = X[test_mask]
                y_val_subj = y[test_mask]

                # internal train/test
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_train_subj, y_train_subj, test_size=0.3,
                    stratify=y_train_subj, random_state=42
                )

                # CSP again
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

                model = get_model(model_type, nb_classes)
                model.fit(X_tr_csp, y_tr)

                int_preds = model.predict(X_te_csp)
                excl_preds = model.predict(X_val_csp)

                internal_acc = accuracy_score(y_te, int_preds)
                internal_recall = recall_score(y_te, int_preds, average="macro", zero_division=0)
                internal_kappa = cohen_kappa_score(y_te, int_preds)

                excl_acc = accuracy_score(y_val_subj, excl_preds)
                excl_recall = recall_score(y_val_subj, excl_preds, average="macro", zero_division=0)
                excl_kappa = cohen_kappa_score(y_val_subj, excl_preds)

                subj_results.append({
                    "model_type": model_type,
                    "label_map": map_name,
                    "subject": subj,
                    "baseline_acc": base_acc,
                    "baseline_recall": base_recall,
                    "baseline_kappa": base_kappa,
                    "internal_acc": internal_acc,
                    "internal_recall": internal_recall,
                    "internal_kappa": internal_kappa,
                    "excluded_acc": excl_acc,
                    "excluded_recall": excl_recall,
                    "excluded_kappa": excl_kappa
                })

            subj_df = pd.DataFrame(subj_results)
            avg_row = {
                "model_type": model_type,
                "label_map": map_name,
                "subject": "average",
                "baseline_acc": subj_df["baseline_acc"].mean(),
                "baseline_recall": subj_df["baseline_recall"].mean(),
                "baseline_kappa": subj_df["baseline_kappa"].mean(),
                "internal_acc": subj_df["internal_acc"].mean(),
                "internal_recall": subj_df["internal_recall"].mean(),
                "internal_kappa": subj_df["internal_kappa"].mean(),
                "excluded_acc": subj_df["excluded_acc"].mean(),
                "excluded_recall": subj_df["excluded_recall"].mean(),
                "excluded_kappa": subj_df["excluded_kappa"].mean(),
            }
            """
            avg_row = {
            "model_type": model_type,
            "label_map": map_name,
            "subject": "average",
            "baseline_acc": base_acc,
            "baseline_recall": base_recall,
            "baseline_kappa": base_kappa,
            }
            all_results.extend(subj_results)
            all_results.append(avg_row)

    input_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = f"model_eval/csp/sk_models_loso_eval_{input_name}_n_components_{n_csp_components}.xlsx"
    pd.DataFrame(all_results).to_excel(output_path, index=False)
    print(f"\nResults saved to {output_path}")
