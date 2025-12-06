# train_rf.py
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from extractor import extract_features

# ---------------------------------------------------
# 1. PRZYGOTOWANIE DANYCH
# ---------------------------------------------------
def load_dataset(real_dir, fake_dir):
    X, y = [], []

    for fname in os.listdir(real_dir):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(real_dir, fname)
            X.append(extract_features(path))
            y.append(0)

    for fname in os.listdir(fake_dir):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(fake_dir, fname)
            X.append(extract_features(path))
            y.append(1)

    return np.array(X), np.array(y)


# ---------------------------------------------------
# 2. TRENING
# ---------------------------------------------------
def train_rf(real_dir, fake_dir, save_path="rf_model.pkl"):
    X, y = load_dataset(real_dir, fake_dir)
    print("Dataset loaded:", X.shape, "samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n=== TEST RESULTS ===")
    print(classification_report(y_test, preds))

    joblib.dump(model, save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    train_rf("data/real", "data/fake")
