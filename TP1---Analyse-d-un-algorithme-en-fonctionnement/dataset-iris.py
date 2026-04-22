import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)
print("=== EXPLORATION DES DONNÉES ===")
print(f"Forme du dataset : {df.shape}")
print(f"\nDistribution des classes :")
print(df['species'].value_counts())
print(f"\nValeurs manquantes : {df.isnull().sum().sum()}")
print(f"\nAperçu :")
print(df.head(3))
print(f"\nStatistiques descriptives :")
print(df.describe())

sns.pairplot(df, hue='species', markers=["o", "s", "D"],
plot_kws=dict(alpha=0.7), diag_kind='hist')
plt.suptitle("Dataset Iris — Séparabilité des classes", y=1.02)
plt.savefig("iris_pairplot.png", bbox_inches="tight")