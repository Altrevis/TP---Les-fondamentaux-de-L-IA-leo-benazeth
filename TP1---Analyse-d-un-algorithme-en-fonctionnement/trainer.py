import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

le = LabelEncoder()
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = le.fit_transform(df['species'])
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# Split entraînement/test (80%/20%) stratifié
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Entraînement : {len(X_train)} échantillons | Test : {len(X_test)} échantillons")
# Normalisation (importante pour comparer les variables sur la même échelle)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
# Baseline simple : Decision Tree (max_depth=3)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train_sc, y_train)
# Modèle principal : Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_sc, y_train)
print("Modèles entraînés.")

y_pred_dt = dt.predict(X_test_sc)
y_pred_rf = rf.predict(X_test_sc)
print("=== COMPARAISON BASELINE vs RANDOM FOREST ===")
for nom, pred in [("Decision Tree (baseline)", y_pred_dt), ("Random Forest ", y_pred_rf)]:
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    print(f"{nom} → Accuracy : {acc*100:.1f}% | F1-score (weighted) : {f1:.3f}")
print("\n=== RAPPORT DÉTAILLÉ — RANDOM FOREST ===")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matrice de confusion — Random Forest')
plt.ylabel('Réalité')
plt.xlabel('Prédiction')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()