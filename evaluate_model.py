"""
Model Evaluation Script for the Autism Gaze Classifier.

Performs cross-validation, computes accuracy/precision/recall/F1,
generates a confusion matrix, and prints a summary for project reports.

Run: python evaluate_model.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from app.classifier import _generate_reference_data, FEATURE_NAMES


def evaluate():
    """Run full model evaluation and print results."""

    print("=" * 60)
    print("  MODEL EVALUATION — Autism Gaze Classifier")
    print("=" * 60)
    print()

    # ── Generate data ──
    X, y = _generate_reference_data(n_typical=200, n_atypical=200, seed=42)
    print(f"📊 Dataset: {len(X)} samples ({sum(y == 0)} typical, {sum(y == 1)} atypical)")
    print(f"   Features: {FEATURE_NAMES}")
    print()

    # ── 1. Cross-Validation (10-fold) ──
    print("─" * 60)
    print("1️⃣  10-FOLD CROSS-VALIDATION")
    print("─" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Multiple metrics
    acc_scores = cross_val_score(clf, X_scaled, y, cv=kfold, scoring="accuracy")
    prec_scores = cross_val_score(clf, X_scaled, y, cv=kfold, scoring="precision")
    rec_scores = cross_val_score(clf, X_scaled, y, cv=kfold, scoring="recall")
    f1_scores = cross_val_score(clf, X_scaled, y, cv=kfold, scoring="f1")
    auc_scores = cross_val_score(clf, X_scaled, y, cv=kfold, scoring="roc_auc")

    print(f"   Accuracy:   {acc_scores.mean():.4f}  (±{acc_scores.std():.4f})")
    print(f"   Precision:  {prec_scores.mean():.4f}  (±{prec_scores.std():.4f})")
    print(f"   Recall:     {rec_scores.mean():.4f}  (±{rec_scores.std():.4f})")
    print(f"   F1 Score:   {f1_scores.mean():.4f}  (±{f1_scores.std():.4f})")
    print(f"   AUC-ROC:    {auc_scores.mean():.4f}  (±{auc_scores.std():.4f})")
    print()

    # ── 2. Train/Test Split (80/20) ──
    print("─" * 60)
    print("2️⃣  TRAIN/TEST SPLIT (80% train, 20% test)")
    print("─" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set:     {len(X_test)} samples")
    print()
    print(f"   Accuracy:     {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Precision:    {precision_score(y_test, y_pred):.4f}")
    print(f"   Recall:       {recall_score(y_test, y_pred):.4f}")
    print(f"   F1 Score:     {f1_score(y_test, y_pred):.4f}")
    print(f"   AUC-ROC:      {roc_auc_score(y_test, y_proba):.4f}")
    print()

    # ── 3. Confusion Matrix ──
    print("─" * 60)
    print("3️⃣  CONFUSION MATRIX")
    print("─" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"                  Predicted")
    print(f"                  Typical  Atypical")
    print(f"   Actual Typical    {cm[0][0]:3d}      {cm[0][1]:3d}")
    print(f"   Actual Atypical   {cm[1][0]:3d}      {cm[1][1]:3d}")
    print()
    tn, fp, fn, tp = cm.ravel()
    print(f"   True Positives:   {tp}  (correctly identified atypical)")
    print(f"   True Negatives:   {tn}  (correctly identified typical)")
    print(f"   False Positives:  {fp}  (typical misclassified as atypical)")
    print(f"   False Negatives:  {fn}  (atypical misclassified as typical)")
    print()

    # ── 4. Classification Report ──
    print("─" * 60)
    print("4️⃣  CLASSIFICATION REPORT")
    print("─" * 60)
    print(classification_report(y_test, y_pred, target_names=["Typical", "Atypical"]))

    # ── 5. Feature Importance ──
    print("─" * 60)
    print("5️⃣  FEATURE IMPORTANCE")
    print("─" * 60)
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        bar = "█" * int(importances[i] * 40)
        print(f"   {FEATURE_NAMES[i]:30s}  {importances[i]:.4f}  {bar}")
    print()

    # ── Summary ──
    print("=" * 60)
    print("  SUMMARY FOR PROJECT REPORT")
    print("=" * 60)
    print(f"""
   Model:             Random Forest (100 trees, max_depth=5)
   Dataset:           {len(X)} synthetic samples (research-based)
   Cross-Val Acc:     {acc_scores.mean():.1%} (±{acc_scores.std():.1%})
   Test Accuracy:     {accuracy_score(y_test, y_pred):.1%}
   AUC-ROC:           {roc_auc_score(y_test, y_proba):.4f}
   
   Top Feature:       {FEATURE_NAMES[sorted_idx[0]]}
   
   Note: Model trained on synthetic data generated from
   published research on gaze patterns in ASD. For clinical
   use, training on real clinical data would be required.
""")
    print("=" * 60)

    # ── Save results to file ──
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/model_evaluation.txt", "w", encoding="utf-8") as f:
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: Random Forest (100 trees, max_depth=5)\n")
        f.write(f"Dataset: {len(X)} synthetic samples\n\n")
        f.write(f"10-Fold Cross-Validation:\n")
        f.write(f"  Accuracy:   {acc_scores.mean():.4f} (±{acc_scores.std():.4f})\n")
        f.write(f"  Precision:  {prec_scores.mean():.4f} (±{prec_scores.std():.4f})\n")
        f.write(f"  Recall:     {rec_scores.mean():.4f} (±{rec_scores.std():.4f})\n")
        f.write(f"  F1 Score:   {f1_scores.mean():.4f} (±{f1_scores.std():.4f})\n")
        f.write(f"  AUC-ROC:    {auc_scores.mean():.4f} (±{auc_scores.std():.4f})\n\n")
        f.write(f"Train/Test Split (80/20):\n")
        f.write(f"  Accuracy:   {accuracy_score(y_test, y_pred):.4f}\n")
        f.write(f"  Precision:  {precision_score(y_test, y_pred):.4f}\n")
        f.write(f"  Recall:     {recall_score(y_test, y_pred):.4f}\n")
        f.write(f"  F1 Score:   {f1_score(y_test, y_pred):.4f}\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TN={tn}  FP={fp}\n")
        f.write(f"  FN={fn}  TP={tp}\n\n")
        f.write(f"Feature Importance:\n")
        for i in sorted_idx:
            f.write(f"  {FEATURE_NAMES[i]:30s}  {importances[i]:.4f}\n")
    print("✅ Results saved to experiments/model_evaluation.txt")


if __name__ == "__main__":
    evaluate()
