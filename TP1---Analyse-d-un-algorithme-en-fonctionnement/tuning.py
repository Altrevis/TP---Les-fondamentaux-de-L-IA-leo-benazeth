from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from trainer import X_test, X_test_sc, X_train, X_train_sc, y_train, y_test
n_estimators_range = [10, 25, 50, 100, 200, 500]
scores_cv = []
for n in n_estimators_range:
	m = RandomForestClassifier(n_estimators=n, random_state=42)
	cv = cross_val_score(m, X_train_sc, y_train, cv=5, scoring='accuracy')
	scores_cv.append(cv.mean())
	print(f"n_estimators={n:4d} → CV accuracy : {cv.mean()*100:.1f}% (±{cv.std()*100:.1f}%)")
profondeurs = range(1, 20)
scores_train = []
scores_test = []
for d in profondeurs:
	m = RandomForestClassifier(n_estimators=50, max_depth=d, random_state=42)
	m.fit(X_train_sc, y_train)
	scores_train.append(m.score(X_train_sc, y_train))
	scores_test.append(m.score(X_test_sc, y_test))
plt.figure(figsize=(9, 4))
plt.plot(profondeurs, scores_train, 'b-o', label='Score entraînement')
plt.plot(profondeurs, scores_test, 'r-o', label='Score test')
plt.xlabel('Profondeur maximale (max_depth)')
plt.ylabel('Accuracy')
plt.title('Underfitting vs Overfitting — Random Forest')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('overfitting.png')
plt.show()