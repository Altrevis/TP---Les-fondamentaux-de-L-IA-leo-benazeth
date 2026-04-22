import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from modeles import modeles, X_train_sc, X_test_sc, y_train, y_test

NOM_MEILLEUR = "Logistic Regression"
meilleur = modeles[NOM_MEILLEUR]
y_pred = meilleur.predict(X_test_sc)

print(f"=== RAPPORT DÉTAILLÉ — {NOM_MEILLEUR} ===")
print(classification_report(y_test, y_pred, target_names=['Non-Churn', 'Churn']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Churn', 'Churn'],
            yticklabels=['Non-Churn', 'Churn'])
plt.title(f'Matrice de confusion — {NOM_MEILLEUR}')
plt.ylabel('Réalité')
plt.xlabel('Prédiction')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

n_estimators_range = [10, 25, 50, 100, 200]
scores_rf = []
for n in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_temp.fit(X_train_sc, y_train)
    pred_temp = rf_temp.predict(X_test_sc)
    scores_rf.append(f1_score(y_test, pred_temp, average='macro'))
    print(f"n_estimators={n:4d} → F1-macro : {scores_rf[-1]:.3f}")

plt.figure(figsize=(8, 4))
plt.plot(n_estimators_range, scores_rf, 'g-o')
plt.xlabel("Nombre d'arbres (n_estimators)")
plt.ylabel("F1-score macro")
plt.title("Évolution des performances — Random Forest")
plt.grid(True)
plt.tight_layout()
plt.savefig('rf_evolution.png')
plt.show()
