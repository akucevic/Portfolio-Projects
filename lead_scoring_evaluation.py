# Lead Scoring Model Evaluation

from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Assuming model and data are already loaded
# model = ...
# X, y = ...

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print("Cross-validated ROC AUC:", cv_scores.mean())

# Precision-recall curve
probs = model.predict_proba(X)[:, 1]
precision, recall, _ = precision_recall_curve(y, probs)

plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# Classification report
y_pred = model.predict(X)
print(classification_report(y, y_pred))
