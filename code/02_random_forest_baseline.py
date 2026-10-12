#Random Forest baseline on clean (unobfuscated) CIC-Darknet2020 traffic-type data,
#matching the classical-ML baseline methodology used in Rust-Nguyen et al. (2023)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import json
import joblib
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project folder (one level above code/)
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze("columns")
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze("columns")

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight="balanced",  # dataset is imbalanced (Tor = 0.98% of rows)
)

print("Training Random Forest...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print(f"\nClean-data accuracy: {acc:.4f}")
print(f"Clean-data macro F1: {f1_macro:.4f}")
print(f"Clean-data weighted F1: {f1_weighted:.4f}")
print("\nClassification report:")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
print("\nConfusion matrix (rows=true, cols=pred):")
print("Classes:", list(clf.classes_))
print(cm)

# Feature importances (for methodology write-up / comparison to Marim et al.)
importances = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(importances.head(10))

# --- Save model and results ---
joblib.dump(clf, f"{RESULTS_DIR}/rf_model.joblib")

results = {
    "model": "RandomForest (n_estimators=200, balanced class weights)",
    "clean_accuracy": float(acc),
    "clean_f1_macro": float(f1_macro),
    "clean_f1_weighted": float(f1_weighted),
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "classes": list(clf.classes_),
    "top_10_features": importances.head(10).to_dict(),
}
with open(f"{RESULTS_DIR}/rf_clean_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved model and results to {RESULTS_DIR}")
