"""
import os
import numpy as np
import pandas as pd
import warnings
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, cohen_kappa_score
)
import xgboost as xgb


def full_pipeline(config: dict) -> None:
    """
"""
    Full pipeline:
    - Preprocess raw EDF → segments
    - Compute PSD features
    - Apply label maps
    - Train + evaluate XGBoost classifier
    - Save results + feature importances
    """
"""
    # ------------------------------
    # Step 1: Preprocessing
    # ------------------------------
    print("== Step 1: Preprocessing raw EEG ==")

    
    processed_path = os.path.join(config["data"]["processed"],
                                  "data_segments_all_channels_combined_2_5secondepoch.npz")
    data = np.load(processed_path, allow_pickle=True)
    X, y, subject = data["X"], data["y"], data["subject"]
    print(f"Loaded preprocessed data: {X.shape}, labels={y.shape}, subjects={subject.shape}")

    # ------------------------------
    # Step 2: PSD features
    # ------------------------------
    print("== Step 2: PSD Feature Extraction ==")
    bands = config["psd_bands"]
    X_psd = compute_psd_features(X, bands, fs=256)  # shape: (epochs, ch*bands)
    print(f"PSD features shape: {X_psd.shape}")

    # ------------------------------
    # Step 3: Classification
    # ------------------------------
    print("== Step 3: Classification ==")

    label_map_configs = {
        "0_vs_2": {0: 0, 1: None, 2: 1},
        "0_vs_1": {0: 0, 1: 1, 2: None},
        "1_vs_2": {0: None, 1: 0, 2: 1},
        "0_vs_1and2": {0: 0, 1: 1, 2: 1},
        "0and1_vs_2": {0: 0, 1: 0, 2: 1}
    }

    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else 0.5
        kappa = cohen_kappa_score(y_true, y_pred)
        return acc, f1, prec, rec, auc, kappa

    results_summary = []
    feature_importance_dfs = {}

    for map_name, label_map in label_map_configs.items():
        print(f"\n--- Running map {map_name} ---")

        # filter labels
        mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
        if mask.sum() == 0:
            print(f"Skipping {map_name}: no data left")
            continue

        X_filtered = X_psd[mask]
        y_filtered = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)

        if len(np.unique(y_filtered)) < 2:
            print(f"Skipping {map_name}: <2 classes")
            continue

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics_list, importances = [], []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_filtered, y_filtered)):
            X_train, X_test = X_filtered[train_idx], X_filtered[test_idx]
            y_train, y_test = y_filtered[train_idx], y_filtered[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = xgb.XGBClassifier(
                n_estimators=1000, learning_rate=0.1,
                objective="binary:logistic", eval_metric="logloss",
                use_label_encoder=False, random_state=42+fold,
                tree_method="hist", device="cuda"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            preds = model.predict(X_test)
            metrics_list.append(get_metrics(y_test, preds))
            importances.append(model.feature_importances_)

        # aggregate metrics
        metrics_arr = np.array(metrics_list)
        means, stds = metrics_arr.mean(axis=0), metrics_arr.std(axis=0)
        metric_names = ["Acc","F1","Prec","Recall","AUROC","Kappa"]

        results_summary.append({
            "label_map": map_name,
            **{f"{n}_mean": m for n, m in zip(metric_names, means)},
            **{f"{n}_std": s for n, s in zip(metric_names, stds)}
        })

        # aggregate importances
        avg_imp = np.mean(importances, axis=0)
        importance_df = pd.DataFrame({
            "feature_idx": np.arange(avg_imp.size),
            "importance": avg_imp
        }).sort_values("importance", ascending=False)
        feature_importance_dfs[map_name] = importance_df

        # save CSV
        imp_out = os.path.join(config["results"]["importance"], f"importance_{map_name}.csv")
        importance_df.to_csv(imp_out, index=False)
        print(f"Saved importance {map_name} -> {imp_out}")

    # save summary
    summary_df = pd.DataFrame(results_summary)
    outpath = os.path.join(config["results"]["summary"], "classification_summary.csv")
    summary_df.to_csv(outpath, index=False)
    print(f"\n== Done. Summary saved to {outpath} ==")
    print(summary_df)
"""