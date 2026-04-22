import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

# Chargement du dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

class_names = [
    'T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
    'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine'
]

# ─────────────────────────────────────────────
# ÉTAPE 2 — Prétraitement des images
# ─────────────────────────────────────────────
# Normalisation : [0, 255] → [0.0, 1.0]
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm  = X_test.astype('float32') / 255.0

print(f"Après normalisation : min={X_train_norm.min():.1f}, max={X_train_norm.max():.1f}")

# Pour scikit-learn : aplatir chaque image 28x28 en vecteur de 784 valeurs
X_train_flat = X_train_norm.reshape(-1, 784)
X_test_flat  = X_test_norm.reshape(-1, 784)

print(f"Forme pour ML classique (aplatie) : {X_train_flat.shape}")
print(f"Forme pour réseau dense (grille)  : {X_train_norm.shape}")

# Pour CNN : ajouter la dimension canal (niveaux de gris = 1 canal)
X_train_cnn = X_train_norm.reshape(-1, 28, 28, 1)
X_test_cnn  = X_test_norm.reshape(-1, 28, 28, 1)
print(f"Forme pour CNN : {X_train_cnn.shape}")

# ─────────────────────────────────────────────
# ÉTAPE 3 — Baseline : Random Forest (ML classique)
# ─────────────────────────────────────────────
print("\n=== ÉTAPE 3 : Random Forest ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_flat, y_train)
y_pred_rf = rf.predict(X_test_flat)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest — Accuracy : {acc_rf*100:.1f}%")

# Sauvegarde
joblib.dump(rf, 'rf_model.pkl')
print("Modèle RF sauvegardé : rf_model.pkl")

# ─────────────────────────────────────────────
# ÉTAPE 4 — Réseau de neurones dense (MLP)
# ─────────────────────────────────────────────
print("\n=== ÉTAPE 4 : Réseau Dense (MLP) ===")
model_dense = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model_dense.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_dense.summary()

history_dense = model_dense.fit(
    X_train_norm, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.15,
    verbose=1
)

# Sauvegarde
model_dense.save('model_dense.keras')
print("Réseau dense sauvegardé : model_dense.keras")

# ─────────────────────────────────────────────
# ÉTAPE 5 — Réseau convolutif (CNN)
# ─────────────────────────────────────────────
print("\n=== ÉTAPE 5 : CNN ===")
model_cnn = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    # Bloc 1 : détection de motifs simples (contours, bords)
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    # Bloc 2 : détection de motifs complexes (formes, structures)
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    # Classification finale
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_cnn.summary()

history_cnn = model_cnn.fit(
    X_train_cnn, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.15,
    verbose=1
)

# Sauvegarde
model_cnn.save('model_cnn.keras')
print("CNN sauvegardé : model_cnn.keras")

# Sauvegarde des historiques d'entraînement pour evaluation.py
histories = {
    'acc_rf': float(acc_rf),
    'dense': {
        'accuracy':     history_dense.history['accuracy'],
        'val_accuracy': history_dense.history['val_accuracy'],
        'loss':         history_dense.history['loss'],
        'val_loss':     history_dense.history['val_loss'],
    },
    'cnn': {
        'accuracy':     history_cnn.history['accuracy'],
        'val_accuracy': history_cnn.history['val_accuracy'],
        'loss':         history_cnn.history['loss'],
        'val_loss':     history_cnn.history['val_loss'],
    }
}

with open('histories.json', 'w') as f:
    json.dump(histories, f)
print("Historiques sauvegardés : histories.json")
print("\nEntraînement terminé. Lancez evaluation.py puis interpretabilite.py.")
