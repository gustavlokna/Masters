import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from mne.decoding import CSP
import os


def load_psd_data(npz_path, config):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    subject = data["subject"]
    bands = data["bands"]
    sex = data["sex"]
    age = data["age"]

    selected = np.array(config["channels"]["top_64"]) - 1
    X = X[:, :, selected]

    return X, y, subject, bands, sex, age


def prepare_data(X, y, label_map):
    mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
    X_filtered = X[mask]
    y_filtered = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)
    return X_filtered, y_filtered


def train_model(model_type, X_train, y_train):
    if model_type == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=len(np.unique(y_train)),
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42,
            tree_method='hist',
            device='cuda'
        )
    elif model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=10)
    elif model_type == "SVC":
        model = SVC(kernel='rbf', probability=True)
    elif model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    elif model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=1000, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_test, preds)
    print(classification_report(y_test, preds, zero_division=0))
    return acc, recall, kappa


def balance_train_classes(X_train, y_train):
    X_df = pd.DataFrame(X_train)
    y_df = pd.Series(y_train, name="label")
    df = pd.concat([X_df, y_df], axis=1)

    majority_class = df["label"].value_counts().idxmax()
    minority_class = df["label"].value_counts().idxmin()

    df_majority = df[df["label"] == majority_class]
    df_minority = df[df["label"] == minority_class]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    X_balanced = df_balanced.drop(columns=["label"]).values
    y_balanced = df_balanced["label"].values

    return X_balanced, y_balanced


def csp_testing(config, file_path):
    X, y, subject, band_names, sex, age = load_psd_data(file_path, config)
    all_results = []

    model_types = ["SVC", "LogisticRegression", "RandomForest", "XGBoost"]

    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Running label map: {map_name} ===")
        subj_filtered = np.array(subject)[[label_map.get(lbl, None) is not None for lbl in y]]

        X_filtered, y_filtered = prepare_data(X, y, label_map)

        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.3, stratify=y_filtered, random_state=42
        )

        # CSP before scaling
        X_train_csp_input = X_train.transpose(0, 2, 1)
        X_test_csp_input = X_test.transpose(0, 2, 1)

        csp = CSP(n_components=64, log=True)
        X_train = csp.fit_transform(X_train_csp_input, y_train)
        X_test = csp.transform(X_test_csp_input)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for model_type in model_types:
            print(f"\n=== {model_type} Baseline Model (70/30 Split) ===")

            model = train_model(model_type, X_train, y_train)
            base_acc, base_recall, base_kappa = evaluate_model(model, X_test, y_test)

            subj_results = []
            subjects = np.unique(subj_filtered)

            for subj in subjects:
                train_mask = np.array([subj not in str(s) for s in subj_filtered])
                test_mask = np.array([str(s) == str(subj) for s in subj_filtered])

                X_train_subj = X_filtered[train_mask]
                y_train_subj = y_filtered[train_mask]
                X_val_subj = X_filtered[test_mask]
                y_val_subj = y_filtered[test_mask]

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_train_subj, y_train_subj, test_size=0.3, stratify=y_train_subj, random_state=42
                )

                # CSP before scaling
                X_tr_csp_input = X_tr.transpose(0, 2, 1)
                X_te_csp_input = X_te.transpose(0, 2, 1)
                X_val_csp_input = X_val_subj.transpose(0, 2, 1)

                csp = CSP(n_components=6, log=True)
                X_tr = csp.fit_transform(X_tr_csp_input, y_tr)
                X_te = csp.transform(X_te_csp_input)
                X_val_subj = csp.transform(X_val_csp_input)

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
                X_val_subj = scaler.transform(X_val_subj)

                model = train_model(model_type, X_tr, y_tr)

                internal_acc, internal_recall, internal_kappa = evaluate_model(model, X_te, y_te)
                excl_acc, excl_recall, excl_kappa = evaluate_model(model, X_val_subj, y_val_subj)

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

            all_results.extend(subj_results)
            all_results.append(avg_row)

    input_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = f"model_eval/csp/csp_eval_{input_name}_loaded.xlsx"

    df_results = pd.DataFrame(all_results)
    df_results.to_excel(output_path, index=False)
    print(f"\nResults saved to {output_path}")
