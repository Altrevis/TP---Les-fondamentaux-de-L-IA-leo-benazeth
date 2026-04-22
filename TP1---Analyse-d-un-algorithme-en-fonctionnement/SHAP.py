import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from trainer import X_train_sc, X_test_sc, y_train, feature_names, le
rf_final = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_final.fit(X_train_sc, y_train)
explainer = shap.TreeExplainer(rf_final)
shap_values = explainer.shap_values(X_test_sc)
plt.figure()
shap.summary_plot(
shap_values, X_test_sc,
feature_names=feature_names,
class_names=list(le.classes_),
plot_type='bar',
show=False
)
plt.title("SHAP — Importance globale des variables (3 classes)")
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.show()
importances_sklearn = rf_final.feature_importances_
if isinstance(shap_values, list):
    importances_shap = np.abs(shap_values[2]).mean(axis=0) 
else:
    importances_shap = np.abs(shap_values[:, :, 2]).mean(axis=0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.barh(feature_names, importances_sklearn, color='steelblue')
ax1.set_title('Feature Importance — sklearn (MDI)')
ax1.set_xlabel('Importance')
ax2.barh(feature_names, importances_shap, color='darkorange')
ax2.set_title('Feature Importance — SHAP (classe virginica)')
ax2.set_xlabel('|SHAP value| moyen')
plt.tight_layout()
plt.savefig('shap_comparison.png')
plt.show()