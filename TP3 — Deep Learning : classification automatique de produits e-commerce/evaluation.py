import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Chargement du dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

class_names = [
    'T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
    'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine'
]

# Prétraitement (identique à entrainement.py)
X_test_norm = X_test.astype('float32') / 255.0
X_test_flat = X_test_norm.reshape(-1, 784)
X_test_cnn  = X_test_norm.reshape(-1, 28, 28, 1)

# Chargement des modèles et historiques sauvegardés
print("Chargement des modèles et historiques...")
rf          = joblib.load('rf_model.pkl')
model_dense = keras.models.load_model('model_dense.keras')
model_cnn   = keras.models.load_model('model_cnn.keras')

with open('histories.json') as f:
    histories = json.load(f)

acc_rf = histories['acc_rf']

# ─────────────────────────────────────────────
# ÉTAPE 6 — Courbes d'apprentissage
# ─────────────────────────────────────────────
print("\n=== ÉTAPE 6 : Courbes d'apprentissage ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Réseau dense
axes[0].plot(histories['dense']['accuracy'],     'b-', label='Train')
axes[0].plot(histories['dense']['val_accuracy'], 'r-', label='Validation')
axes[0].set_title('Réseau Dense (MLP)')
axes[0].set_xlabel('Époque')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# CNN
axes[1].plot(histories['cnn']['accuracy'],     'b-', label='Train')
axes[1].plot(histories['cnn']['val_accuracy'], 'r-', label='Validation')
axes[1].set_title('CNN')
axes[1].set_xlabel('Époque')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Courbes d'apprentissage — Diagnostic overfitting", fontsize=13)
plt.tight_layout()
plt.savefig('learning_curves.png')
plt.show()
print("Courbes sauvegardées : learning_curves.png")

# ─────────────────────────────────────────────
# ÉTAPE 7 — Comparaison des 3 approches
# ─────────────────────────────────────────────
print("\n=== ÉTAPE 7 : Comparaison des 3 approches ===")

loss_dense, acc_dense = model_dense.evaluate(X_test_norm, y_test, verbose=0)
loss_cnn,   acc_cnn   = model_cnn.evaluate(X_test_cnn,  y_test, verbose=0)

print("=" * 55)
print("  COMPARAISON DES 3 APPROCHES — COMITÉ TECHNIQUE")
print("=" * 55)
print(f"  {'Modèle':<20s} {'Accuracy':>10s} {'Taux erreur':>12s}")
print("-" * 55)
print(f"  {'Random Forest':<20s} {acc_rf*100:>9.1f}% {(1-acc_rf)*100:>11.1f}%")
print(f"  {'Réseau Dense':<20s} {acc_dense*100:>9.1f}% {(1-acc_dense)*100:>11.1f}%")
print(f"  {'CNN':<20s} {acc_cnn*100:>9.1f}% {(1-acc_cnn)*100:>11.1f}%")
print("-" * 55)
print(f"  Objectif business : taux d'erreur < 5.0%")
print("=" * 55)

# Prédictions du CNN pour l'analyse détaillée
y_pred_cnn     = model_cnn.predict(X_test_cnn, verbose=0)
y_pred_classes = np.argmax(y_pred_cnn, axis=1)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matrice de confusion — CNN (articles Zalando)')
plt.ylabel('Catégorie réelle')
plt.xlabel('Catégorie prédite')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn.png')
plt.show()
print("Matrice sauvegardée : confusion_matrix_cnn.png")

print("\n=== RAPPORT PAR CATÉGORIE — CNN ===")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# ─────────────────────────────────────────────
# ÉTAPE 8 — Analyse des erreurs
# ─────────────────────────────────────────────
print("\n=== ÉTAPE 8 : Analyse des erreurs ===")

errors = np.where(y_pred_classes != y_test)[0]
print(f"Erreurs : {len(errors)} / {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)")

# Afficher 10 erreurs types
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    idx = errors[i]
    ax.imshow(X_test[idx], cmap='gray')
    confiance = y_pred_cnn[idx, y_pred_classes[idx]] * 100
    ax.set_title(
        f"Prédit : {class_names[y_pred_classes[idx]]} ({confiance:.0f}%)\n"
        f"Réel   : {class_names[y_test[idx]]}",
        fontsize=8, color='red'
    )
    ax.axis('off')

plt.suptitle('CNN — Articles mal classifiés (analyse qualité)', fontsize=13)
plt.tight_layout()
plt.savefig('erreurs_cnn.png')
plt.show()
print("Erreurs sauvegardées : erreurs_cnn.png")

# Top 5 confusions les plus fréquentes
print("\n=== TOP 5 CONFUSIONS LES PLUS FRÉQUENTES ===")
confusions = {}
for real, pred in zip(y_test[errors], y_pred_classes[errors]):
    pair = (class_names[real], class_names[pred])
    confusions[pair] = confusions.get(pair, 0) + 1

top_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:5]
for (real, pred), count in top_confusions:
    print(f"  {real:12s} → classifié comme {pred:12s} : {count} erreurs")
