import numpy as np
import matplotlib.pyplot as plt
import shap
import scipy.sparse as sp
from modeles import modeles, X_test_sc, feature_names

rf_model = modeles["Random Forest"]

X_sample = X_test_sc[:200]
if sp.issparse(X_sample):
    X_sample = X_sample.toarray()
else:
    X_sample = np.array(X_sample)

explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    sv = shap_values[1]               
elif len(shap_values.shape) == 3:
    sv = shap_values[:, :, 1]         
else:
    sv = shap_values                  

plt.figure()
shap.summary_plot(
    sv, X_sample,
    feature_names=feature_names,
    max_display=15,
    show=False
)
plt.title("SHAP — Top 15 variables les plus influentes (Churn = 1)")
plt.tight_layout()
plt.savefig('shap_summary.png', bbox_inches='tight')
plt.show()

importances_shap = np.abs(sv).mean(axis=0)
top_idx = np.argsort(importances_shap)[::-1][:10]
print("=== TOP 10 VARIABLES SHAP ===")
for i in top_idx:
    print(f"  {feature_names[i]:40s} : {importances_shap[i]:.4f}")
