from mne.decoding import CSP
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score, classification_report
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
from utilities.EEGNet import DeepConvNet

# Enable GPU usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {tf.config.get_visible_devices('GPU')[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU.")


def load_raw_data(npz_path, config):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    subject = data["subject"]
    sex = data["sex"]
    age = data["age"]

    print(f"Loaded RAW data: X={X.shape}, y={y.shape}")

    #selected = np.array(config["channels"]["top_64"]) - 1
    #X = np.transpose(X, (0, 2, 1))
    X = np.expand_dims(X, axis=-1)

    return X, y, subject, sex, age


def prepare_data(X, y, label_map):
    mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
    X_filtered = X[mask]
    y_filtered = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)
    return X_filtered, y_filtered


def balance_classes(X, y):
    y_labels = np.argmax(y, axis=1)
    unique, counts = np.unique(y_labels, return_counts=True)
    if len(unique) < 2:
        return X, y
    max_count = counts.max()
    X_balanced, y_balanced = [], []
    for cls in unique:
        X_cls = X[y_labels == cls]
        y_cls = y[y_labels == cls]
        X_res, y_res = resample(X_cls, y_cls, replace=True, n_samples=max_count, random_state=42)
        X_balanced.append(X_res)
        y_balanced.append(y_res)
    return np.vstack(X_balanced), np.vstack(y_balanced)


def train_deep_eegnet(X_train, y_train, nb_classes):
    chans = X_train.shape[1]
    samples = X_train.shape[2]
    model = DeepConvNet(nb_classes=nb_classes, Chans=chans, Samples=samples, dropoutRate=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    print(classification_report(y_true, y_pred, zero_division=0))
    return acc, recall, kappa



def csp_testing(config, file_path):
    X_raw, y_raw, subject, sex, age = load_raw_data(file_path, config)

    all_results = []
    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Running label map: {map_name} ===")

        X, y = prepare_data(X_raw, y_raw, label_map)
        subj_filtered = np.array(subject)[[label_map.get(lbl, None) is not None for lbl in y_raw]]
        subjects = np.unique(subj_filtered)

        nb_classes = len(np.unique(y))
        y_cat = to_categorical(y, nb_classes)

        # CSP: reshape (samples, channels, time, 1) -> (samples, channels, time)
        X_csp_input = X.squeeze(-1)
        X_csp_input = X_csp_input.transpose(0, 2, 1)  # to (samples, time, channels)

        n_csp_components = 64 
        csp = CSP(n_components=n_csp_components, log=True)
        X_csp = csp.fit_transform(X_csp_input, y)

        # reshape CSP output back into (samples, fake_channels, time, 1)
        X_csp = X_csp.reshape(X_csp.shape[0], n_csp_components, 1, 1)

        # baseline split
        X_train, X_test, y_train, y_test = train_test_split(
            X_csp, to_categorical(y, nb_classes), test_size=0.3, stratify=y, random_state=42
        )
        X_train, y_train = balance_classes(X_train, y_train)

        # flatten then scale
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

        print("=== Training baseline ===")
        model = train_deep_eegnet(X_train_scaled, y_train, nb_classes)
        base_acc, base_recall, base_kappa = evaluate_model(model, X_test_scaled, y_test)

        subj_results = []
        for subj in np.unique(subjects):
            train_mask = np.array([subj not in str(s) for s in subj_filtered])
            test_mask = np.array([str(s) == str(subj) for s in subj_filtered])

            X_train_subj = X[train_mask]
            y_train_subj = y_cat[train_mask]
            X_val_subj = X[test_mask]
            y_val_subj = y_cat[test_mask]

            y_train_subj_labels = np.argmax(y_train_subj, axis=1)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train_subj, y_train_subj, test_size=0.3,
                stratify=y_train_subj_labels, random_state=42
            )
            X_tr, y_tr = balance_classes(X_tr, y_tr)

            # CSP on subject-wise training
            X_tr_csp_input = X_tr.squeeze(-1).transpose(0, 2, 1)
            X_te_csp_input = X_te.squeeze(-1).transpose(0, 2, 1)
            X_val_csp_input = X_val_subj.squeeze(-1).transpose(0, 2, 1)

            csp = CSP(n_components=n_csp_components, log=True)
            X_tr_csp = csp.fit_transform(X_tr_csp_input, np.argmax(y_tr, axis=1))
            X_te_csp = csp.transform(X_te_csp_input)
            X_val_csp = csp.transform(X_val_csp_input)

            # Reshape CSP outputs
            X_tr_csp = X_tr_csp.reshape(X_tr_csp.shape[0], n_csp_components, 1, 1)
            X_te_csp = X_te_csp.reshape(X_te_csp.shape[0], n_csp_components, 1, 1)
            X_val_csp = X_val_csp.reshape(X_val_csp.shape[0], n_csp_components, 1, 1)

            # Scaling
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_csp.reshape(X_tr_csp.shape[0], -1)).reshape(X_tr_csp.shape)
            X_te_scaled = scaler.transform(X_te_csp.reshape(X_te_csp.shape[0], -1)).reshape(X_te_csp.shape)
            X_val_scaled = scaler.transform(X_val_csp.reshape(X_val_csp.shape[0], -1)).reshape(X_val_csp.shape)

            model = train_deep_eegnet(X_tr_scaled, y_tr, nb_classes)
            internal_acc, internal_recall, internal_kappa = evaluate_model(model, X_te_scaled, y_te)
            excl_acc, excl_recall, excl_kappa = evaluate_model(model, X_val_scaled, y_val_subj)

            subj_results.append({
                "model_type": "DeepConvNet+CSP",
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

        # average across subjects
        subj_df = pd.DataFrame(subj_results)
        avg_row = {
            "model_type": "DeepConvNet+CSP",
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
    output_path = f"model_eval/loso_eval_{input_name}_with_csp.xlsx"
    pd.DataFrame(all_results).to_excel(output_path, index=False)
    print(f"\nResults saved to {output_path}")
