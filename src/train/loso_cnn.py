import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical


def load_psd_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X, y, subject, bands, sex = data["X"], data["y"], data["subject"], data["bands"], data["sex"]
    print(f"Loaded PSD data: X={X.shape}, y={y.shape}, bands={bands}")
    return X, y, subject, bands, sex


def prepare_data(X, y, subject, sex, label_map):
    n_epochs, n_bands, n_channels = X.shape
    mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
    X_filtered = X[mask]                          # beholder 3D (epochs, bands, channels)
    y_filtered = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)
    subject_filtered = subject[mask]
    sex_filtered = sex[mask]
    return X_filtered, y_filtered, subject_filtered, sex_filtered


def build_cnn_lstm(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def leave_one_subject_out(X, y, subject_ids, sex_ids, map_name, epochs=10, batch_size=32):
    results = []
    unique_subjects = np.unique(subject_ids)
    num_classes = len(np.unique(y))

    for subj in unique_subjects:
        X_train = X[subject_ids != subj]
        y_train = y[subject_ids != subj]
        X_test = X[subject_ids == subj]
        y_test = y[subject_ids == subj]
        sex_subj = np.unique(sex_ids[subject_ids == subj])[0]

        # one-hot labels
        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_test_cat = to_categorical(y_test, num_classes=num_classes)

        input_shape = (X.shape[1], X.shape[2])
        model = build_cnn_lstm(input_shape, num_classes)
        model.fit(X_train, y_train_cat, epochs=epochs, batch_size=batch_size,
                  verbose=0, validation_data=(X_test, y_test_cat))

        preds = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_test, preds, zero_division=0, output_dict=True)

        pred_0 = int(np.sum(preds == 0))
        pred_1 = int(np.sum(preds == 1))
        true_0 = int(np.sum(y_test == 0))
        true_1 = int(np.sum(y_test == 1))

        row = {
            "label_map": map_name,
            "subject": subj,
            "sex": sex_subj,
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1": report["macro avg"]["f1-score"],
            "support": report["macro avg"]["support"],
            "pred_0": pred_0,
            "true_0": true_0,
            "pred_1": pred_1,
            "true_1": true_1,
        }
        results.append(row)

    return results


def loso_pipeline(config, excel_out="loso_cnn_results.xlsx"):
    X, y, subject, band_names, sex = load_psd_data(config["data"]["psd"])
    all_results = []

    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Running label map: {map_name} ===")
        X_filtered, y_filtered, subject_filtered, sex_filtered = prepare_data(X, y, subject, sex, label_map)

        if len(np.unique(y_filtered)) < 2:
            print(f"Skipping {map_name} (not enough classes).")
            continue

        results = leave_one_subject_out(
            X_filtered, y_filtered, subject_filtered, sex_filtered, map_name,
            epochs=200,
            batch_size=32
        )
        all_results.extend(results)
        print(f"Done with LOSO for {map_name}.")

    df = pd.DataFrame(all_results)
    df.to_excel(excel_out, index=False)
    print(f"\n✅ Results saved to {excel_out}")
