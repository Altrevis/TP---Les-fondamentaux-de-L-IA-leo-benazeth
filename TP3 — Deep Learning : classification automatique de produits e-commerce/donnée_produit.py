import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
class_names = [
'T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine'
]
print("=== EXPLORATION DU CATALOGUE ===")
print(f"Train : {X_train.shape[0]} images de {X_train.shape[1]}x{X_train.shape[2]} pixels")
print(f"Test : {X_test.shape[0]} images")
print(f"Pixels : min={X_train.min()}, max={X_train.max()} (niveaux de gris 0–255)")
print(f"Catégories : {len(class_names)}")
print("\nDistribution des catégories (train) :")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f" {class_names[u]:12s} : {c:>5d} articles ({c/len(y_train)*100:.1f}%)")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(class_names[i], fontsize=10)
    ax.axis('off')
plt.suptitle('Catalogue Zalando — 1 exemple par catégorie', fontsize=13)
plt.tight_layout()
plt.savefig('catalogue_samples.png')
plt.show()