# train_ai_vs_real.py

import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from PIL import Image
import clip
import numpy as np

def load_data(split="train"):
    ds = load_dataset("Parveshiiii/AI-vs-Real", split=split, verification_mode = "no_checks")
    # Upewnij się, że kolumna "image" jest w formacie PIL
    ds = ds.cast_column("image", ds.features["image"])
    return ds

def preprocess_and_embed(ds, model, preprocess, device):
    embeddings = []
    labels = []
    for example in ds:
        img: Image.Image = example["image"]
        label = example["binary_label"]
        img = img.convert("RGB")
        inp = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(inp)
            emb = emb.cpu().numpy().flatten()
        embeddings.append(emb)
        labels.append(label)
    X = np.stack(embeddings)
    y = np.array(labels)
    return X, y

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. Ładowanie danych
    ds = load_data("train")
    print("Loaded dataset length:", len(ds))

    # 2. Ładowanie CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # 3. Wyciąganie embeddingów
    print("Extracting embeddings...")
    X, y = preprocess_and_embed(ds, model, preprocess, device)
    print("Embeddings shape:", X.shape)

    # 4. Podział na train / val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 5. Trenowanie prostego klasyfikatora (np. logistic regression)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    # 6. Ewaluacja
    from sklearn.metrics import classification_report, accuracy_score
    y_pred = clf.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred, digits=4))

    # 7. (Opcjonalnie) zapis modelu
    import joblib
    os.makedirs("output", exist_ok=True)
    joblib.dump(clf, "output/clip_lr_ai_vs_real.pkl")
    print("Saved classifier to output/clip_lr_ai_vs_real.pkl")

if __name__ == "__main__":
    main()
