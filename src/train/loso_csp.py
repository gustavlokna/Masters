import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, Reshape


def load_csp_data(npz_path):
    """Load CSP features + labels from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    X, y, subject, sex = data["X"], data["y"], data["subject"], data["sex"]
    print(f"Loaded CSP data: X={X.shape}, y={y.shape}")
    return X, y, subject, sex


def cnn_model(input_shape, X_train, y_train, X_test, y_test):
    """Define and compile a simple CNN+LSTM model."""
    model = Sequential()
    model.add(Reshape((input_shape[0], 1), input_shape=(input_shape[0],)))
    model.add(Conv1D(filters=32, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    return (model.predict(X_test) > 0.5).astype(int).flatten()


def model_xgboost(X_train, y_train, X_test):
    """Train and predict using XGBoost."""
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model.predict(X_test)


def prepare_data(X, y, subject, sex, label_map):
    """Filter CSP features and remap labels according to label_map."""
    mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
    X_filtered = X[mask]
    y_filtered = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)
    subject_filtered = subject[mask]
    sex_filtered = sex[mask]

    return X_filtered, y_filtered, subject_filtered, sex_filtered


def leave_one_subject_out(X, y, subject_ids, sex_ids, map_name):
    """Train on all other subjects, test on one subject, save results to DataFrame."""
    results = []
    unique_subjects = np.unique(subject_ids)

    for subj in unique_subjects:
        sex_subj = np.unique(sex_ids[subject_ids == subj])[0]

        train_mask = (subject_ids != subj) & (sex_ids == sex_subj)
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[subject_ids == subj]
        y_test = y[subject_ids == subj]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Choose model: CNN or XGBoost
        # preds = cnn_model(X_train.shape[1:], X_train, y_train, X_test, y_test)
        preds = model_xgboost(X_train, y_train, X_test)

        report = classification_report(y_test, preds, zero_division=0, output_dict=True)

        row = {
            "label_map": map_name,
            "subject": subj,
            "sex": sex_subj,
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1": report["macro avg"]["f1-score"],
            "support": report["macro avg"]["support"],
            "pred_0": int(np.sum(preds == 0)),
            "true_0": int(np.sum(y_test == 0)),
            "pred_1": int(np.sum(preds == 1)),
            "true_1": int(np.sum(y_test == 1)),
        }
        results.append(row)

    return results


def loso_psd_pipeline(config, excel_out="loso_results_csp_data.xlsx"):
    """Main entry point for LOSO training pipeline on CSP data."""
    # Just read directly from config
    csp_path = config["data"]["csp"]  # or config["data"]["csp_fbcsp"] for FBCSP
    X, y, subject, sex = load_csp_data(csp_path)
    all_results = []

    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Running label map: {map_name} ===")
        X_filtered, y_filtered, subject_filtered, sex_filtered = prepare_data(X, y, subject, sex, label_map)

        if len(np.unique(y_filtered)) < 2:
            print(f"Skipping {map_name} (not enough classes).")
            continue

        results = leave_one_subject_out(X_filtered, y_filtered, subject_filtered, sex_filtered, map_name)
        all_results.extend(results)
        print(f"Done with LOSO for {map_name}.")

    df = pd.DataFrame(all_results)
    df.to_excel(excel_out, index=False)
    print(f"\n✅ Results saved to {excel_out}")
