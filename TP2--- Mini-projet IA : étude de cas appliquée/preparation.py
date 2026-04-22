import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("=== EXPLORATION DES DONNÉES ===")
print(f"Forme : {df.shape}")
print(f"\nDistribution cible :")
print(df['Churn'].value_counts())
print(f"\nColonnes : {list(df.columns)}")
print(f"\nValeurs manquantes par colonne :")
print(df.isnull().sum()[df.isnull().sum() > 0])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
print(f"\nAprès nettoyage : {df.shape}")

df_enc = pd.get_dummies(df.drop('customerID', axis=1), drop_first=True)
X = df_enc.drop('Churn_Yes', axis=1)
y = df_enc['Churn_Yes'].astype(int)
feature_names = list(X.columns)

print(f"\nTaux de churn : {y.mean()*100:.1f}% → dataset déséquilibré")
print("→ Utiliser F1-macro plutôt qu'accuracy seule")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")
print(f"Nombre de features : {len(feature_names)}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, col in zip(axes, ['tenure', 'MonthlyCharges', 'TotalCharges']):
    df.groupby('Churn')[col].plot(kind='hist', alpha=0.6, ax=ax, legend=True)
    ax.set_title(col)
    ax.set_xlabel(col)
plt.suptitle("Distribution des variables clés selon le Churn")
plt.tight_layout()
plt.savefig('distribution_variables.png')
plt.show()
