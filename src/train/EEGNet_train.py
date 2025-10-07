import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from utilities.EEGNet import EEGNet

# --- GPU setup ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected. Running on CPU.")


def load_eeg_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X, y, subject, sex = data["X"], data["y"], data["subject"], data["sex"]
    print(f"Loaded EEG data: X={X.shape}, y={y.shape}, sex={sex.shape}")
    return X, y, subject, sex



def prepare_for_eegnet(X, y):
    # Reshape to EEGNet format
    X = np.transpose(X, (0, 2, 1))   # (N, chans, samples)
    X = X[..., np.newaxis]           # (N, chans, samples, 1)

    n_classes = len(np.unique(y))
    Y = to_categorical(y, num_classes=n_classes)
    return X, Y, n_classes


def run_eegnet(X, Y, n_classes, epochs=50, batch_size=32):
    # Split: 70/20/10
    X_train, X_temp, Y_train, Y_temp, y_train, y_temp = train_test_split(
        X, Y, Y.argmax(axis=-1), test_size=0.3, stratify=Y.argmax(axis=-1), random_state=42
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=1/3, stratify=y_temp, random_state=42
    )

    # Build EEGNet model
    model = EEGNet(nb_classes=n_classes, Chans=X.shape[1], Samples=X.shape[2])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Fit
    model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, Y_val),
        verbose=2
    )

    # Predict
    preds = model.predict(X_test)
    pred_labels = preds.argmax(axis=-1)
    true_labels = Y_test.argmax(axis=-1)

    print("\n=== Classification Report (Test Set) ===")
    print(classification_report(true_labels, pred_labels, zero_division=0))



def run_eegnet_pipeline(config):
    X, y, subject, sex = load_eeg_data(config["data"]["preprocessed"])

    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Running label map: {map_name} ===")

        # Mask + remap labels
        mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
        X_filtered = X[mask]
        y_filtered = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)

        if len(np.unique(y_filtered)) < 2:
            print(f"Skipping {map_name} (not enough classes).")
            continue

        # Reshape for EEGNet
        X_ready, Y_ready, n_classes = prepare_for_eegnet(X_filtered, y_filtered)

        # Train + evaluate
        run_eegnet(
            X_ready, Y_ready, n_classes,
            epochs=300,
            batch_size=64
        )
