import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                              ConfusionMatrixDisplay, RocCurveDisplay,
                              precision_recall_curve, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibrationDisplay
from sklearn.utils.class_weight import compute_sample_weight

DATA = "/Users/milesrizor/.cache/kagglehub/datasets/aasheesh200/framingham-heart-study-dataset/versions/1/framingham.csv"

df = pd.read_csv(DATA)

FEATURES = [
    "male", "age", "education", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
]
TARGET = "TenYearCHD"

X = df[FEATURES].values
y = df[TARGET].values

# ── Cross-validation ───────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Logistic Regression with balanced class weights ────────────────────────────
lr_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
])
lr_probs = cross_val_predict(lr_pipe, X, y, cv=cv, method="predict_proba")[:, 1]

# ── Gradient Boosting: manual CV loop for correct per-fold sample weights ──────
# GradientBoostingClassifier doesn't support class_weight; passing sample_weight
# via cross_val_predict fit_params doesn't subset by fold, so we loop manually.
gb_probs       = np.zeros(len(y))
gb_importances = np.zeros(len(FEATURES))

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train         = y[train_idx]

    sw  = compute_sample_weight("balanced", y_train)
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_train)
    X_te = imp.transform(X_test)

    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                     learning_rate=0.05, random_state=42)
    gb.fit(X_tr, y_train, sample_weight=sw)
    gb_probs[test_idx]  = gb.predict_proba(X_te)[:, 1]
    gb_importances     += gb.feature_importances_

gb_importances /= 5

# ── Ensemble: average LR + GB probabilities ────────────────────────────────────
ens_probs = (lr_probs + gb_probs) / 2

# ── Threshold tuning: maximize F2 score (recall weighted 2x over precision) ────
# In a screening context a missed CHD case is costlier than a false alarm.
def best_threshold(y_true, probs, beta=2):
    prec, rec, thresholds = precision_recall_curve(y_true, probs)
    denom = beta**2 * prec + rec
    fbeta = np.where(denom > 0, (1 + beta**2) * prec * rec / denom, 0)
    idx = np.argmax(fbeta[:-1])
    return float(thresholds[idx])

lr_thresh  = best_threshold(y, lr_probs)
gb_thresh  = best_threshold(y, gb_probs)
ens_thresh = best_threshold(y, ens_probs)

lr_pred  = (lr_probs  >= lr_thresh).astype(int)
gb_pred  = (gb_probs  >= gb_thresh).astype(int)
ens_pred = (ens_probs >= ens_thresh).astype(int)
baseline_pred = np.zeros(len(y), dtype=int)

# ── Metrics ────────────────────────────────────────────────────────────────────
def report(name, y_true, y_pred, y_prob, threshold):
    acc        = accuracy_score(y_true, y_pred)
    auc        = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens       = tp / (tp + fn)
    spec       = tn / (tn + fp)
    print(f"{name:<25} Acc={acc*100:.1f}%  AUC={auc:.3f}  "
          f"Sensitivity={sens*100:.1f}%  Specificity={spec*100:.1f}%  "
          f"Threshold={threshold:.2f}")
    return acc, auc, sens, spec

print("── 5-fold CV results (F2-tuned thresholds) ─────────────────────────────")
base_acc = accuracy_score(y, baseline_pred)
print(f"{'Always-No-CHD baseline':<25} Acc={base_acc*100:.1f}%  AUC=0.500  "
      f"Sensitivity=0.0%   Specificity=100.0%")
lr_acc,  lr_auc,  lr_sens,  lr_spec  = report("Logistic Regression",  y, lr_pred,  lr_probs,  lr_thresh)
gb_acc,  gb_auc,  gb_sens,  gb_spec  = report("Gradient Boosting",    y, gb_pred,  gb_probs,  gb_thresh)
ens_acc, ens_auc, ens_sens, ens_spec = report("Ensemble",             y, ens_pred, ens_probs, ens_thresh)

# ── Feature importances for display ───────────────────────────────────────────
feat_imp = pd.Series(gb_importances, index=FEATURES).sort_values()

lr_pipe.fit(X, y)
lr_coefs = pd.Series(
    np.abs(lr_pipe.named_steps["model"].coef_[0]), index=FEATURES
).sort_values()

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. ROC curves — all three models
ax_roc = fig.add_subplot(gs[0, 0])
for probs, name, col in [
    (lr_probs,  f"Logistic Reg (AUC={lr_auc:.2f})",   "#2a6fa8"),
    (gb_probs,  f"Gradient Boost (AUC={gb_auc:.2f})", "#e07b39"),
    (ens_probs, f"Ensemble (AUC={ens_auc:.2f})",       "#2ca02c"),
]:
    RocCurveDisplay.from_predictions(y, probs, name=name, ax=ax_roc, color=col)
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax_roc.set_title("ROC Curve (5-fold CV)")
ax_roc.legend(fontsize=7)

# 2. Precision-Recall curve with tuned threshold marked
ax_pr = fig.add_subplot(gs[0, 1])
for probs, thresh, name, col in [
    (lr_probs,  lr_thresh,  "Logistic Reg",  "#2a6fa8"),
    (gb_probs,  gb_thresh,  "Gradient Boost","#e07b39"),
    (ens_probs, ens_thresh, "Ensemble",      "#2ca02c"),
]:
    prec, rec, _ = precision_recall_curve(y, probs)
    ax_pr.plot(rec, prec, label=name, color=col)
    pred = (probs >= thresh).astype(int)
    ax_pr.scatter([recall_score(y, pred)], [precision_score(y, pred)],
                  color=col, s=60, zorder=5)
ax_pr.axhline(y.mean(), color="k", linestyle="--", linewidth=0.8,
              label=f"No-skill ({y.mean()*100:.1f}%)")
ax_pr.set_xlabel("Recall (Sensitivity)")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall (● = tuned threshold)")
ax_pr.legend(fontsize=7)

# 3. Calibration curves
ax_cal = fig.add_subplot(gs[0, 2])
for probs, name, col in [
    (lr_probs,  "Logistic Reg",  "#2a6fa8"),
    (gb_probs,  "Gradient Boost","#e07b39"),
    (ens_probs, "Ensemble",      "#2ca02c"),
]:
    CalibrationDisplay.from_predictions(y, probs, n_bins=10,
                                         name=name, ax=ax_cal, color=col)
ax_cal.set_title("Calibration Curve\n(closer to diagonal = better)")
ax_cal.legend(fontsize=7)

# 4. Confusion matrix — Ensemble at tuned threshold
ax_cm = fig.add_subplot(gs[1, 0])
ConfusionMatrixDisplay(confusion_matrix(y, ens_pred),
                       display_labels=["No CHD", "CHD"]).plot(ax=ax_cm, colorbar=False)
ax_cm.set_title(f"Confusion Matrix — Ensemble\n(threshold={ens_thresh:.2f}, F2-tuned)")

# 5. Metric comparison — all four models
ax_bar = fig.add_subplot(gs[1, 1])
metric_labels = ["Accuracy", "AUC", "Sensitivity", "Specificity"]
vals = {
    "Baseline":    [base_acc, 0.5,     0.0,      1.0],
    "Log Reg":     [lr_acc,   lr_auc,  lr_sens,  lr_spec],
    "Grad Boost":  [gb_acc,   gb_auc,  gb_sens,  gb_spec],
    "Ensemble":    [ens_acc,  ens_auc, ens_sens, ens_spec],
}
colors = ["#aac4e0", "#2a6fa8", "#e07b39", "#2ca02c"]
x = np.arange(len(metric_labels))
w = 0.18
for i, (label, v) in enumerate(vals.items()):
    ax_bar.bar(x + (i - 1.5) * w, v, w, label=label, color=colors[i])
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metric_labels, fontsize=8)
ax_bar.set_ylim(0, 1.1)
ax_bar.set_ylabel("Score")
ax_bar.set_title("Metric Comparison (tuned thresholds)")
ax_bar.legend(fontsize=7)

# 6. Feature importances (GB, averaged across folds)
ax_fi = fig.add_subplot(gs[1, 2])
feat_imp.plot.barh(ax=ax_fi, color="#e07b39")
ax_fi.set_title("Feature Importances\n(Gradient Boosting, avg. across folds)")
ax_fi.set_xlabel("Importance")

plt.suptitle("Framingham Heart Study — 10-Year CHD Risk Prediction (v2)",
             fontsize=13, y=1.01)
plt.savefig("/Users/milesrizor/summer-claude/framingham_results.png",
            dpi=150, bbox_inches="tight")
print("\nPlot saved.")
