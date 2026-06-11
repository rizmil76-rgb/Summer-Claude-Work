import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                              ConfusionMatrixDisplay, RocCurveDisplay,
                              precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA = "/Users/milesrizor/.cache/kagglehub/datasets/aasheesh200/framingham-heart-study-dataset/versions/1/framingham.csv"

df = pd.read_csv(DATA)

FEATURES = [
    "male", "age", "education", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
]
TARGET = "TenYearCHD"

X = df[FEATURES]
y = df[TARGET]

# ── Models ────────────────────────────────────────────────────────────────────
# Logistic regression: interpretable, good clinical baseline
# Gradient boosting: captures non-linear interactions
lr_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000, random_state=42)),
])

gb_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("model",  GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                           learning_rate=0.05, random_state=42)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# cross_val_predict gives out-of-fold predictions → honest evaluation, no leakage
lr_probs = cross_val_predict(lr_pipe, X, y, cv=cv, method="predict_proba")[:, 1]
gb_probs = cross_val_predict(gb_pipe, X, y, cv=cv, method="predict_proba")[:, 1]

# Default 0.5 threshold predictions
lr_pred = (lr_probs >= 0.5).astype(int)
gb_pred = (gb_probs >= 0.5).astype(int)
baseline_pred = np.zeros(len(y), dtype=int)   # always predict "no CHD"

# ── Metrics ────────────────────────────────────────────────────────────────────
def report(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn)   # recall for CHD-positive (sensitivity)
    spec = tn / (tn + fp)   # recall for CHD-negative (specificity)
    print(f"{name:<25} Acc={acc*100:.1f}%  AUC={auc:.3f}  Sensitivity={sens*100:.1f}%  Specificity={spec*100:.1f}%")
    return acc, auc, sens, spec

print("── 5-fold cross-validated results ──────────────────────────────────────")
base_acc = accuracy_score(y, baseline_pred)
print(f"{'Always-No-CHD baseline':<25} Acc={base_acc*100:.1f}%  AUC=0.500  Sensitivity=0.0%   Specificity=100.0%")
lr_acc, lr_auc, lr_sens, lr_spec = report("Logistic Regression",   y, lr_pred, lr_probs)
gb_acc, gb_auc, gb_sens, gb_spec = report("Gradient Boosting",     y, gb_pred, gb_probs)

# ── Feature importances (fit once on full data for display only) ──────────────
gb_pipe.fit(X, y)
feat_imp = pd.Series(
    gb_pipe.named_steps["model"].feature_importances_, index=FEATURES
).sort_values()

lr_pipe.fit(X, y)
lr_coefs = pd.Series(
    np.abs(lr_pipe.named_steps["model"].coef_[0]), index=FEATURES
).sort_values()

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. ROC curves
ax_roc = fig.add_subplot(gs[0, 0])
RocCurveDisplay.from_predictions(y, lr_probs, name="Logistic Reg", ax=ax_roc, color="#2a6fa8")
RocCurveDisplay.from_predictions(y, gb_probs, name="Gradient Boost", ax=ax_roc, color="#e07b39")
ax_roc.plot([0,1],[0,1],"k--", linewidth=0.8, label="Random (AUC=0.5)")
ax_roc.set_title("ROC Curve (5-fold CV)")
ax_roc.legend(fontsize=8)

# 2. Precision-Recall curve (more honest for imbalanced data)
ax_pr = fig.add_subplot(gs[0, 1])
for probs, name, col in [(lr_probs, "Logistic Reg", "#2a6fa8"),
                          (gb_probs, "Gradient Boost", "#e07b39")]:
    prec, rec, _ = precision_recall_curve(y, probs)
    ax_pr.plot(rec, prec, label=name, color=col)
ax_pr.axhline(y.mean(), color="k", linestyle="--", linewidth=0.8,
              label=f"No-skill ({y.mean()*100:.1f}%)")
ax_pr.set_xlabel("Recall (Sensitivity)")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall Curve")
ax_pr.legend(fontsize=8)

# 3. Summary bar chart
ax_bar = fig.add_subplot(gs[0, 2])
metrics = ["Accuracy", "AUC", "Sensitivity", "Specificity"]
baseline_vals = [base_acc, 0.5, 0.0, 1.0]
lr_vals  = [lr_acc, lr_auc, lr_sens, lr_spec]
gb_vals  = [gb_acc, gb_auc, gb_sens, gb_spec]
x = np.arange(len(metrics))
w = 0.25
ax_bar.bar(x - w, baseline_vals, w, label="Baseline", color="#aac4e0")
ax_bar.bar(x,     lr_vals,  w, label="Log Reg",  color="#2a6fa8")
ax_bar.bar(x + w, gb_vals,  w, label="Grad Boost", color="#e07b39")
ax_bar.set_xticks(x); ax_bar.set_xticklabels(metrics, fontsize=8)
ax_bar.set_ylim(0, 1.05); ax_bar.set_ylabel("Score")
ax_bar.set_title("Metric Comparison")
ax_bar.legend(fontsize=8)

# 4. Confusion matrix — Gradient Boosting
ax_cm = fig.add_subplot(gs[1, 0])
ConfusionMatrixDisplay(confusion_matrix(y, gb_pred),
                       display_labels=["No CHD", "CHD"]).plot(ax=ax_cm, colorbar=False)
ax_cm.set_title("Confusion Matrix\n(Gradient Boosting, threshold=0.5)")

# 5. GB feature importances
ax_fi = fig.add_subplot(gs[1, 1])
feat_imp.plot.barh(ax=ax_fi, color="#e07b39")
ax_fi.set_title("Feature Importances\n(Gradient Boosting)")
ax_fi.set_xlabel("Importance")

# 6. Logistic regression coefficients (absolute)
ax_lr = fig.add_subplot(gs[1, 2])
lr_coefs.plot.barh(ax=ax_lr, color="#2a6fa8")
ax_lr.set_title("Feature Coefficients |β|\n(Logistic Regression)")
ax_lr.set_xlabel("|Coefficient|")

plt.suptitle("Framingham Heart Study — 10-Year CHD Risk Prediction", fontsize=13, y=1.01)
plt.savefig("/Users/milesrizor/summer-claude/framingham_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved.")
