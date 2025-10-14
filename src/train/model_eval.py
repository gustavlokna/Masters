import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import xgboost as xgb


def model_mlp(input_dim, X_train, y_train, X_test):
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0)
    return (model.predict(X_test) > 0.5).astype(int).flatten()


def load_psd_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data["X"], data["y"], data["subject"], data["bands"], data["sex"], data["age"]


def prepare_data(X, y, label_map):
    n_epochs, n_bands, n_channels = X.shape
    X_flat = X.reshape(n_epochs, n_bands * n_channels)
    mask = np.array([label_map.get(lbl, None) is not None for lbl in y])
    X_filtered = X_flat[mask]
    y_filtered = np.array([label_map[lbl] for lbl in y[mask]], dtype=int)
    return X_filtered, y_filtered


def train_model(model_type, X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    if model_type == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.1, objective='binary:logistic',
            eval_metric='logloss', use_label_encoder=False, random_state=42,
            tree_method='hist', device='cuda'
        )
        model.fit(X_train, y_train)

    elif model_type == "MLP":
        return "MLP", scaler  # Placeholder (handled separately)

    elif model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=10) #was 5

    elif model_type == "SVC":
        model = SVC(kernel='rbf', probability=True)

    elif model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)

    elif model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=1000, random_state=42)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type != "XGBoost":
        model.fit(X_train, y_train)

    return model, scaler


def evaluate_model(model, scaler, X_test, y_test, model_type, X_train=None, y_train=None):
    X_test_scaled = scaler.transform(X_test)

    if model_type == "MLP":
        preds = model_mlp(X_train.shape[1], X_train, y_train, X_test_scaled)
    else:
        preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_test, preds)
    print(classification_report(y_test, preds, zero_division=0))
    return acc, recall, kappa

def add_sex_age_features(X, sex, age):
    """
    Add sex and age as extra features to each epoch.

    Parameters
    ----------
    X : ndarray, shape (epochs, bands, channels)
        PSD features.
    sex : ndarray, shape (epochs,)
        Binary or categorical sex values (e.g., 0=female, 1=male).
    age : ndarray, shape (epochs,)
        Age values per epoch.

    Returns
    -------
    X_out : ndarray, shape (epochs, bands, channels + 2)
        PSD features with sex and age appended as two additional channels.
    """
    X = np.asarray(X)
    sex = np.asarray(sex).reshape(-1, 1, 1)
    age = np.asarray(age).reshape(-1, 1, 1)

    sex_expanded = np.repeat(sex, X.shape[1], axis=1)   # match number of bands
    age_expanded = np.repeat(age, X.shape[1], axis=1)

    X_out = np.concatenate((X, sex_expanded, age_expanded), axis=2)
    return X_out

def balance_train_classes(X_train, y_train):
    """
    Balance class distribution in the training set by upsampling the minority class.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Feature matrix for the training data.
    y_train : array-like, shape (n_samples,)
        Labels corresponding to X_train.

    Returns
    -------
    X_balanced : ndarray, shape (n_samples_balanced, n_features)
        Training feature matrix with balanced class representation.
    y_balanced : ndarray, shape (n_samples_balanced,)
        Labels corresponding to X_balanced with equal class counts.

    Notes
    -----
    - The function identifies the minority and majority classes in `y_train`.
    - The minority class is upsampled with replacement until it matches
      the number of samples in the majority class.
    - The resulting dataset preserves all majority samples and duplicates
      minority samples as needed.
    """
    X_df = pd.DataFrame(X_train)
    y_df = pd.Series(y_train, name="label")
    df = pd.concat([X_df, y_df], axis=1)

    majority_class = df["label"].value_counts().idxmax()
    minority_class = df["label"].value_counts().idxmin()

    df_majority = df[df["label"] == majority_class]
    df_minority = df[df["label"] == minority_class]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    X_balanced = df_balanced.drop(columns=["label"]).values
    y_balanced = df_balanced["label"].values

    return X_balanced, y_balanced

def models_eval(config, output_excel="all_models_all_labels_with_sex_and_age.xlsx"):
    X, y, subject, band_names, sex ,age= load_psd_data(config["data"]["psd"])
    X = add_sex_age_features(X, sex, age) 
    all_results = []
     #["MLP", "KNN", "SVC", "LogisticRegression", "RandomForest", "XGBoost"]
    model_types = ["MLP", "KNN", "SVC", "LogisticRegression", "RandomForest", "XGBoost"]

    for map_name, label_map in config["label_maps"].items():
        print(f"\n=== Running label map: {map_name} ===")
        X_filtered, y_filtered = prepare_data(X, y, label_map)
        subj_filtered = np.array(subject)[[label_map.get(lbl, None) is not None for lbl in y]]

        if len(np.unique(y_filtered)) < 2:
            print(f"Skipping {map_name} (not enough classes).")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.3, stratify=y_filtered, random_state=42
        )
        #X_train, y_train = balance_train_classes(X_train, y_train)

        for model_type in model_types:
            print(f"\n=== {model_type} Baseline Model (70/30 Split) ===")
            model, scaler = train_model(model_type, X_train, y_train)

            if model_type == "MLP":
                base_acc, base_recall, base_kappa = evaluate_model(
                    model, scaler, X_test, y_test, model_type, X_train, y_train
                )
            else:
                base_acc, base_recall, base_kappa = evaluate_model(
                    model, scaler, X_test, y_test, model_type
                )

            subjects = np.unique(subj_filtered)

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

                model, scaler = train_model(model_type, X_tr, y_tr)

                if model_type == "MLP":
                    internal_acc, internal_recall, internal_kappa = evaluate_model(
                        model, scaler, X_te, y_te, model_type, X_tr, y_tr
                    )
                    excl_acc, excl_recall, excl_kappa = evaluate_model(
                        model, scaler, X_val_subj, y_val_subj, model_type, X_tr, y_tr
                    )
                else:
                    internal_acc, internal_recall, internal_kappa = evaluate_model(
                        model, scaler, X_te, y_te, model_type
                    )
                    excl_acc, excl_recall, excl_kappa = evaluate_model(
                        model, scaler, X_val_subj, y_val_subj, model_type
                    )

                all_results.append({
                    "model_type": model_type,
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

                print(f"{model_type} -> Subject {subj}: internal_acc={internal_acc:.4f}, excluded_acc={excl_acc:.4f}")

    df_results = pd.DataFrame(all_results)
    df_results.to_excel(output_excel, index=False)
    print(f"\nResults saved to {output_excel}")
