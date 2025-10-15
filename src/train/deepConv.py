import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score, classification_report
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
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

    selected = np.array(config["channels"]["top_64"]) - 1
    X = X[:, :, selected]
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


def train_eegnet(X_train, y_train, nb_classes):
    chans = X_train.shape[1]
    samples = X_train.shape[2]
    model = DeepConvNet(nb_classes=nb_classes, Chans=chans, Samples=samples, dropoutRate=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=300, batch_size=32, verbose=1)
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


def test_deep_conv(config, output_excel="deep_conv_3_labels_raw_top64_results.xlsx"):
    X, y, subject, sex, age = load_raw_data(config["data"]["preprocessed"], config)
    all_results = []

    nb_classes = len(np.unique(y))
    y_cat = to_categorical(y, nb_classes)

    # 70/30 split baseline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.3, stratify=y, random_state=42
    )
    X_train, y_train = balance_classes(X_train, y_train)
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
    model = train_eegnet(X_train, y_train, nb_classes)
    base_acc, base_recall, base_kappa = evaluate_model(model, X_test, y_test)
    print(f"Baseline -> Acc: {base_acc:.4f}, Recall: {base_recall:.4f}, Kappa: {base_kappa:.4f}")
    """
    subjects = np.unique(subject)
    
    for subj in subjects:
        train_mask = subject != subj
        test_mask = subject == subj

        X_train_subj = X[train_mask]
        y_train_subj = y_cat[train_mask]
        X_val_subj = X[test_mask]
        y_val_subj = y_cat[test_mask]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train_subj, y_train_subj, test_size=0.3,
            stratify=np.argmax(y_train_subj, axis=1), random_state=42
        )

        X_tr, y_tr = balance_classes(X_tr, y_tr)

        model = train_eegnet(X_tr, y_tr, nb_classes)
        internal_acc, internal_recall, internal_kappa = evaluate_model(model, X_te, y_te)
        excl_acc, excl_recall, excl_kappa = evaluate_model(model, X_val_subj, y_val_subj)

        all_results.append({
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
    """
    all_results.append({
        "baseline_acc": base_acc,
        "baseline_recall": base_recall,
        "baseline_kappa": base_kappa,
    })
    df_results = pd.DataFrame(all_results)
    df_results.to_excel(output_excel, index=False)
    print(f"\nResults saved to {output_excel}")




def eeg_loso(config, output_excel="EEGNet_loso_raw_channel_reduction_top64_results.xlsx"):
    X, y, subject, sex, age = load_raw_data(config["data"]["preprocessed"], config)
    all_results = []

    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Running label map: {map_name} ===")
        X_filtered, y_filtered = prepare_data(X, y, label_map)
        subj_filtered = np.array(subject)[[label_map.get(lbl, None) is not None for lbl in y]]

        if len(np.unique(y_filtered)) < 2:
            print(f"Skipping {map_name} (not enough classes).")
            continue

        nb_classes = len(np.unique(y_filtered))
        y_cat = to_categorical(y_filtered, nb_classes)

        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_cat, test_size=0.3, stratify=y_filtered, random_state=42
        )
        X_train, y_train = balance_classes(X_train, y_train)

        print("\n=== Baseline EEGNet (70/30 Split) ===")
        model = train_eegnet(X_train, y_train, nb_classes)
        base_acc, base_recall, base_kappa = evaluate_model(model, X_test, y_test)
        print(f"Baseline -> Acc: {base_acc:.4f}, Recall: {base_recall:.4f}, Kappa: {base_kappa:.4f}")

        subjects = np.unique(subj_filtered)

        for subj in subjects:
            train_mask = subj_filtered != subj
            test_mask = subj_filtered == subj

            X_train_subj = X_filtered[train_mask]
            y_train_subj = y_cat[train_mask]
            X_val_subj = X_filtered[test_mask]
            y_val_subj = y_cat[test_mask]

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train_subj, y_train_subj, test_size=0.3,
                stratify=np.argmax(y_train_subj, axis=1), random_state=42
            )

            X_tr, y_tr = balance_classes(X_tr, y_tr)

            model = train_eegnet(X_tr, y_tr, nb_classes)
            internal_acc, internal_recall, internal_kappa = evaluate_model(model, X_te, y_te)
            excl_acc, excl_recall, excl_kappa = evaluate_model(model, X_val_subj, y_val_subj)

            all_results.append({
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

            print(f"Subject {subj}: internal_acc={internal_acc:.4f}, excluded_acc={excl_acc:.4f}")

    df_results = pd.DataFrame(all_results)
    df_results.to_excel(output_excel, index=False)
    print(f"\nResults saved to {output_excel}")
