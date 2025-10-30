import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from utilities.EEGNet import DeepConvNet
import os


def train_deep_eegnet(X_train, y_train, nb_classes):
    chans = X_train.shape[1]
    samples = X_train.shape[2]
    model = DeepConvNet(nb_classes=nb_classes, Chans=chans, Samples=samples, dropoutRate=0.1)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(1e-3), metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    print(classification_report(y_true, y_pred, zero_division=0))
    return acc, recall, kappa


def test_deep_conv_subject(config, data_tuple, subject_id):
    X, y, subject, sex, age = data_tuple

    # shape (samples, samples, channels) → (samples, channels, time, 1)
    X = np.transpose(X, (0, 2, 1))
    X = np.expand_dims(X, axis=-1)

    nb_classes = len(np.unique(y))
    y_cat = to_categorical(y, nb_classes)
    all_results = []

    # LOSO masks
    train_mask = subject != subject_id
    test_mask = subject == subject_id

    X_train_subj = X[train_mask]
    y_train_subj = y_cat[train_mask]
    X_val_subj = X[test_mask]
    y_val_subj = y_cat[test_mask]

    # internal split inside training set
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train_subj, y_train_subj, test_size=0.3,
        stratify=np.argmax(y_train_subj, axis=1), random_state=42
    )

    # standardize per-subject
    nsamp_tr, nch, ntime, _ = X_tr.shape
    nsamp_te = X_te.shape[0]
    nsamp_val = X_val_subj.shape[0]

    scaler = StandardScaler()
    X_tr_flat = X_tr.reshape(nsamp_tr, nch * ntime)
    X_te_flat = X_te.reshape(nsamp_te, nch * ntime)
    X_val_flat = X_val_subj.reshape(nsamp_val, nch * ntime)

    X_tr_scaled = scaler.fit_transform(X_tr_flat).reshape(nsamp_tr, nch, ntime, 1)
    X_te_scaled = scaler.transform(X_te_flat).reshape(nsamp_te, nch, ntime, 1)
    X_val_scaled = scaler.transform(X_val_flat).reshape(nsamp_val, nch, ntime, 1)

    model = train_deep_eegnet(X_tr_scaled, y_tr, nb_classes)

    internal_acc, internal_recall, internal_kappa = evaluate_model(model, X_te_scaled, y_te)
    excl_acc, excl_recall, excl_kappa = evaluate_model(model, X_val_scaled, y_val_subj)

    res = {
        "subject": subject_id,
        "internal_acc": internal_acc,
        "internal_recall": internal_recall,
        "internal_kappa": internal_kappa,
        "excluded_acc": excl_acc,
        "excluded_recall": excl_recall,
        "excluded_kappa": excl_kappa
    }

    all_results.append(res)

    out_dir = "model_eval/memd_loso_deepconv"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"deepconv_eval_{subject_id}.csv")
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
