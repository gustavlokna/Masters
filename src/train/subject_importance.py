import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    cohen_kappa_score,
    classification_report,
)
import xgboost as xgb


def load_psd_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X, y, subject, bands, sex = data["X"], data["y"], data["subject"], data["bands"], data["sex"]
    print(f"Loaded PSD data: X={X.shape}, y={y.shape}, bands={bands}")
    return X, y, subject, bands, sex


def prepare_data(X, y, label_map):
    n_epochs, n_bands, n_channels = X.shape
    X_flat = X.reshape(n_epochs, n_bands * n_channels)
    mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
    X_filtered = X_flat[mask]
    y_filtered = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)
    return X_filtered, y_filtered


def train_model(X_train, y_train):
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
    X_test = scaler.transform(X_test)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_test, preds)
    print(classification_report(y_test, preds, zero_division=0))
    return acc, recall, kappa


def leave_one_subject_out_pipeline(config, output_excel="Marta_loso_results.xlsx"):
    X, y, subject, band_names, sex = load_psd_data(config["data"]["psd"])
    all_results = []

    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Running label map: {map_name} ===")
        X_filtered, y_filtered = prepare_data(X, y, label_map)
        subj_filtered = np.array(subject)[[label_map.get(lbl, None) is not None for lbl in y]]

        if len(np.unique(y_filtered)) < 2:
            print(f"Skipping {map_name} (not enough classes).")
            continue

        # Baseline model (70/30)
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.3, stratify=y_filtered, random_state=42
        )
        print("\n=== Baseline Model (70/30 Split) ===")
        model, scaler = train_model(X_train, y_train)
        base_acc, base_recall, base_kappa = evaluate_model(model, scaler, X_test, y_test)
        print(f"Baseline -> Acc: {base_acc:.4f}, Recall: {base_recall:.4f}, Kappa: {base_kappa:.4f}")

        # Leave-One-Subject-Out evaluation
        subjects = np.unique(subj_filtered)
        subject_results = []

        for subj in subjects:
            train_mask = subj_filtered != subj
            test_mask = subj_filtered == subj

            X_train_subj = X_filtered[train_mask]
            y_train_subj = y_filtered[train_mask]
            X_val_subj = X_filtered[test_mask]
            y_val_subj = y_filtered[test_mask]

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train_subj, y_train_subj, test_size=0.3, stratify=y_train_subj, random_state=42
            )
            model, scaler = train_model(X_tr, y_tr)
            internal_acc, internal_recall, internal_kappa = evaluate_model(model, scaler, X_te, y_te)
            excl_acc, excl_recall, excl_kappa = evaluate_model(model, scaler, X_val_subj, y_val_subj)

            subject_results.append({
                "subject": subj,
                "internal_acc": internal_acc,
                "internal_recall": internal_recall,
                "internal_kappa": internal_kappa,
                "excluded_acc": excl_acc,
                "excluded_recall": excl_recall,
                "excluded_kappa": excl_kappa
            })

            print(f"Subject {subj}: internal_acc={internal_acc:.4f}, excluded_acc={excl_acc:.4f}")

        df_results = pd.DataFrame(subject_results)
        print("\n=== Leave-One-Subject-Out Summary ===")
        print(df_results)

        mean_internal = df_results[["internal_acc", "internal_recall", "internal_kappa"]].mean()
        mean_excluded = df_results[["excluded_acc", "excluded_recall", "excluded_kappa"]].mean()

        all_results.append({
            "label_map": map_name,
            "baseline_acc": base_acc,
            "baseline_recall": base_recall,
            "baseline_kappa": base_kappa,
            "mean_internal_acc": mean_internal["internal_acc"],
            "mean_internal_recall": mean_internal["internal_recall"],
            "mean_internal_kappa": mean_internal["internal_kappa"],
            "mean_excluded_acc": mean_excluded["excluded_acc"],
            "mean_excluded_recall": mean_excluded["excluded_recall"],
            "mean_excluded_kappa": mean_excluded["excluded_kappa"],
        })

    # Save to Excel
    df_all = pd.DataFrame(all_results)
    df_all.to_excel(output_excel, index=False)
    print(f"\nResults saved to {output_excel}")
