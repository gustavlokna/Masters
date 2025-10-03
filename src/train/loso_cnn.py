import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------- IO ----------
def load_psd_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    subject = data["subject"]
    bands = list(data["bands"])
    # optional: sex may or may not exist
    sex = data["sex"] if "sex" in data.files else None
    print(f"Loaded PSD: X={X.shape} (epochs, bands, channels), y={y.shape}, bands={bands}")
    return X, y, subject, bands, sex

def prepare_data(X, y, subject, label_map):
    mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
    Xf = X[mask]                         # (epochs, bands, channels)
    yf = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)
    subjf = subject[mask]
    return Xf, yf, subjf

# ---------- Model ----------
def build_seq_model(input_shape):
    # input_shape = (timesteps=256, features=4)
    m = Sequential([
        Conv1D(64, kernel_size=7, padding="same", activation="relu", input_shape=input_shape),
        Conv1D(64, kernel_size=7, padding="same", activation="relu"),
        LSTM(64),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m

def standardize_train_test(X_train, X_test):
    # per-feature standardization over epochs
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mu) / sd, (X_test - mu) / sd

# ---------- LOSO ----------
def leave_one_subject_out(X, y, subject_ids, map_name, epochs=30, batch_size=64, verbose=0):
    # reshape to (epochs, timesteps=256, features=4): channels-as-time
    # current X is (epochs, bands=4, channels=256)
    X = np.transpose(X, (0, 2, 1))  # -> (epochs, 256, 4)

    unique_subjects = np.unique(subject_ids)

    for subj in unique_subjects:
        tr = subject_ids != subj
        te = subject_ids == subj

        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        # skip if train or test is single-class (can’t learn/measure)
        if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
            print(f"[{map_name}] skip subject {subj}: single-class split (train {np.unique(y_tr)}, test {np.unique(y_te)})")
            continue

        # standardize on train stats
        X_tr, X_te = standardize_train_test(X_tr, X_te)

        # class weights to reduce constant predictions on imbalanced splits
        classes = np.array([0, 1])
        cw_vals = compute_class_weight("balanced", classes=classes, y=y_tr)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}

        model = build_seq_model(input_shape=(X_tr.shape[1], X_tr.shape[2]))
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=verbose,
                  class_weight=class_weight, validation_data=(X_te, y_te))

        preds = (model.predict(X_te, verbose=0).ravel() >= 0.5).astype(int)
        print(f"\n[{map_name}] Subject {subj}")
        print(classification_report(y_te, preds, zero_division=0))

# ---------- Entry ----------
def loso_cnn_pipeline(config):
    X, y, subject, band_names, _ = load_psd_data(config["data"]["psd"])
    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Label map: {map_name} ===")
        Xf, yf, subjf = prepare_data(X, y, subject, label_map)
        if np.unique(yf).size < 2:
            print(f"skip {map_name}: not enough classes after mapping")
            continue
        leave_one_subject_out(Xf, yf, subjf, map_name,
                              epochs= 30,
                              batch_size= 64,
                              verbose= 0)
