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
import os



def load_psd_data(npz_path, config):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]          # (epochs, bands, channels)
    y = data["y"]
    subject = data["subject"]
    bands = data["bands"]
    sex = data["sex"]
    age = data["age"]

    selected = np.array(config["channels"]["top_64"]) - 1
    X = X[:, :, selected]   # select channels along last axis ONLY

    return X, y, subject, bands, sex, age



def prepare_data(X, y):
    n_epochs, n_bands, n_channels = X.shape
    X_flat = X.reshape(n_epochs, n_bands * n_channels)
    return X_flat, y


def train_model(model_type, X_train, y_train):
    if model_type == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=len(np.unique(y_train)),
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42,
            tree_method='hist',
            device='cuda'
        )

    elif model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=10) #was 5

    elif model_type == "SVC":
        model = SVC(kernel='rbf', probability=True)

    elif model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, multi_class='multinomial')

    elif model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=1000, random_state=42)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type != "XGBoost":
        model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
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

def SVC_tetsing(config, file_path):
    X, y, subject, band_names, sex ,age= load_psd_data(file_path, config)
    #X = add_sex_age_features(X, sex, age) 
    all_results = []

    #["KNN", "SVC", "LogisticRegression", "RandomForest", "XGBoost"]
    model_types = ["SVC"]

    X_filtered, y_filtered = prepare_data(X, y)


    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.3, stratify=y_filtered, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #X_train, y_train = balance_train_classes(X_train, y_train)

    for model_type in model_types:
        print(f"\n=== {model_type} Baseline Model (70/30 Split) ===")
        model = train_model(model_type, X_train, y_train)

        base_acc, base_recall, base_kappa = evaluate_model(
            model, X_test, y_test
        )

        subjects = np.unique(subject)
        subj_results = []
        for subj in subjects:
            train_mask = np.array([subj not in str(s) for s in subject]) # written this difficulte to have the opertunity to exclude synthetic samples
            test_mask = np.array([str(s) == str(subj) for s in subject])

            X_train_subj = X_filtered[train_mask]
            y_train_subj = y_filtered[train_mask]
            X_val_subj = X_filtered[test_mask]
            y_val_subj = y_filtered[test_mask]

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train_subj, y_train_subj, test_size=0.3, stratify=y_train_subj, random_state=42
            )
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            X_val_subj = scaler.transform(X_val_subj)

            model = train_model(model_type, X_tr, y_tr)

            internal_acc, internal_recall, internal_kappa = evaluate_model(
                model, X_te, y_te
            )
            excl_acc, excl_recall, excl_kappa = evaluate_model(
                model, X_val_subj, y_val_subj
            )

            subj_results.append({
                "model_type": model_type,
                "label_map": "1,2,3",
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
            "model_type": model_type,
            "label_map": "1,2,3",
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

    # name output file after input npz
    input_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = f"model_eval/standard_models/loso_eval_{input_name}_loaded.xlsx"


    df_results = pd.DataFrame(all_results)
    df_results.to_excel(output_path, index=False)
    print(f"\nResults saved to {output_path}")
