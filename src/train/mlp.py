import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def run_mlp_on_psd(config):
    # Load PSD data
    data = np.load(config["data"]["psd"])
    X = data["X"]
    y = data["y"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize using only training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define MLP model
    model = Sequential()
    model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Train
    model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred_binary)
    print(f"Test accuracy: {acc:.4f}")
