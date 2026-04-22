import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Chargement du dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

class_names = [
    'T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
    'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine'
]

# Prétraitement
X_test_norm = X_test.astype('float32') / 255.0
X_test_cnn  = X_test_norm.reshape(-1, 28, 28, 1)

# Chargement du CNN
print("Chargement du CNN...")
model_cnn = keras.models.load_model('model_cnn.keras')

# ─────────────────────────────────────────────
# ÉTAPE 9a — Filtres appris par la première couche
# ─────────────────────────────────────────────
print("\n=== ÉTAPE 9a : Filtres de la 1re couche Conv2D ===")

first_conv_layer = model_cnn.layers[0]
filters, biases = first_conv_layer.get_weights()
print(f"Filtres : {filters.shape}")  # (3, 3, 1, 32)

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[:, :, 0, i], cmap='gray')
    ax.set_title(f'F{i+1}', fontsize=7)
    ax.axis('off')

plt.suptitle('Filtres appris par le CNN — 1re couche Conv2D (3x3)', fontsize=13)
plt.tight_layout()
plt.savefig('filtres_conv.png')
plt.show()
print("Filtres sauvegardés : filtres_conv.png")

# ─────────────────────────────────────────────
# ÉTAPE 9b — Ce que « voit » le CNN sur un article
# ─────────────────────────────────────────────
print("\n=== ÉTAPE 9b : Activations de la 1re couche ===")

# Modèle intermédiaire — sortie de la 1re couche Conv2D
activation_model = keras.Model(
    inputs=model_cnn.inputs,
    outputs=model_cnn.layers[0].output
)

# Choisir un Sneaker (catégorie 7)
sample_idx = np.where(y_test == 7)[0][0]
sample     = X_test_cnn[sample_idx:sample_idx+1]
activations = activation_model.predict(sample, verbose=0)
print(f"Activations : {activations.shape}")  # (1, 26, 26, 32)

# Image originale + 8 feature maps
fig, axes = plt.subplots(1, 9, figsize=(16, 2.5))
axes[0].imshow(X_test[sample_idx], cmap='gray')
axes[0].set_title('Original', fontsize=9)
axes[0].axis('off')

for i in range(8):
    axes[i+1].imshow(activations[0, :, :, i], cmap='viridis')
    axes[i+1].set_title(f'Filtre {i+1}', fontsize=9)
    axes[i+1].axis('off')

plt.suptitle(f'Activations du CNN — {class_names[y_test[sample_idx]]}', fontsize=12)
plt.tight_layout()
plt.savefig('activations_conv.png')
plt.show()
print("Activations sauvegardées : activations_conv.png")
print("\nÉtape 9 terminée. Tous les livrables sont générés.")
