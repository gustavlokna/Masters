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
    #X = X[:, :, selected]
    X = np.transpose(X, (0, 2, 1))
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
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1) #epochs was 100?
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


def test_deep_conv(config, file_path):
    X, y, subject, sex, age = load_raw_data(file_path, config)

    all_results = []

    nb_classes = len(np.unique(y))
    y_cat = to_categorical(y, nb_classes)

    # 70/30 split baseline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.3, stratify=y, random_state=42
    )
    #X_train, y_train = balance_classes(X_train, y_train)
    # Z-normalization (no data leakage)
    scaler = StandardScaler()
    nsamp, nch, ntime, _ = X_train.shape

    X_train_flat = X_train.reshape(nsamp, nch * ntime)
    X_test_flat = X_test.reshape(X_test.shape[0], nch * ntime)

    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    X_train = X_train_scaled.reshape(nsamp, nch, ntime, 1)
    X_test = X_test_scaled.reshape(X_test.shape[0], nch, ntime, 1)

    print("\n=== Baseline EEGNet (70/30 Split) ===")
    model = train_deep_eegnet(X_train, y_train, nb_classes)
    base_acc, base_recall, base_kappa = evaluate_model(model, X_test, y_test)
    print(f"Baseline -> Acc: {base_acc:.4f}, Recall: {base_recall:.4f}, Kappa: {base_kappa:.4f}")
    """
    subjects = np.unique(subject)
    subj_results = []
    for subj in subjects:
        train_mask = np.array([subj not in str(s) for s in subject]) # written this difficulte to have the opertunity to exclude synthetic samples
        test_mask = np.array([str(s) == str(subj) for s in subject])

        X_train_subj = X[train_mask]
        y_train_subj = y_cat[train_mask]
        X_val_subj = X[test_mask]
        y_val_subj = y_cat[test_mask]

        # internal train/test for the current subject
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train_subj, y_train_subj, test_size=0.3,
            stratify=np.argmax(y_train_subj, axis=1), random_state=42
        )

        X_tr, y_tr = balance_classes(X_tr, y_tr)

        # ---- SUBJECT-WISE SCALING ----
        nsamp_tr, nch, ntime, _ = X_tr.shape
        nsamp_te = X_te.shape[0]
        nsamp_val = X_val_subj.shape[0]

        X_tr_flat = X_tr.reshape(nsamp_tr, nch * ntime)
        X_te_flat = X_te.reshape(nsamp_te, nch * ntime)
        X_val_flat = X_val_subj.reshape(nsamp_val, nch * ntime)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_flat)
        X_te_scaled = scaler.transform(X_te_flat)
        X_val_scaled = scaler.transform(X_val_flat)

        X_tr = X_tr_scaled.reshape(nsamp_tr, nch, ntime, 1)
        X_te = X_te_scaled.reshape(nsamp_te, nch, ntime, 1)
        X_val_subj = X_val_scaled.reshape(nsamp_val, nch, ntime, 1)
        # ---- END SCALING ----

        model = train_deep_eegnet(X_tr, y_tr, nb_classes)
        internal_acc, internal_recall, internal_kappa = evaluate_model(model, X_te, y_te)
        excl_acc, excl_recall, excl_kappa = evaluate_model(model, X_val_subj, y_val_subj)

        subj_results.append({
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

        print(f"Subject {subj}: internal_acc={internal_acc:.4f}, excluded_acc={excl_acc:.4f}")
    
    # average across subjects
    subj_df = pd.DataFrame(subj_results)
    avg_row = {
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
    """
    avg_row = {
        "subject": "average",
        "baseline_acc": base_acc,
        "baseline_recall": base_recall,
        "baseline_kappa": base_kappa,
    }

    all_results.append(avg_row)
    # name output file after input npz
    input_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = f"model_eval/deep_conv_all_chans_{input_name}_loaded.xlsx"

    df_results = pd.DataFrame(all_results)
    df_results.to_excel(output_path, index=False)
    print(f"\nResults saved to {output_path}")