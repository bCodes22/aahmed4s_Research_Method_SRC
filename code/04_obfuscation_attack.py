"""
Replicates the obfuscation-attack methodology from Rust-Nguyen et al. (2023),
adapted from their application-type setting to this project's traffic-type
classification task.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project folder (one level above code/)
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze("columns")
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze("columns")
ip_train = pd.read_csv(f"{DATA_DIR}/ip_train.csv")
ip_test = pd.read_csv(f"{DATA_DIR}/ip_test.csv")
feature_cols = list(X_train.columns)

# --- Define the attacker-controllable "shape" feature subset ---
SHAPE_FEATURES = [c for c in feature_cols if any(k in c for k in [
    "Packet Length", "IAT", "Flow Duration", "Bytes/s", "Packets/s",
    "Average Packet Size", "Segment Size", "Active", "Idle",
])]
print(f"Attacker-controllable 'shape' features ({len(SHAPE_FEATURES)}):")
print(SHAPE_FEATURES)

DARKNET_CLASSES = ["Tor", "VPN", "NonVPN"]
BENIGN_CLASS = "Non-Tor"
BLEND_ALPHA = 0.6  # fraction blended toward benign distribution
NOISE_STD_FRAC = 0.05  # added Gaussian noise, as a fraction of each feature's std

benign_mean = X_train.loc[y_train == BENIGN_CLASS, SHAPE_FEATURES].mean()
feature_std = X_train[SHAPE_FEATURES].std().replace(0, 1e-6)


def obfuscate(X, y):
    """Blend darknet-class rows' shape features toward the benign mean + noise."""
    X_obf = X.copy().astype(float)
    mask = y.isin(DARKNET_CLASSES)
    n_affected = mask.sum()
    noise = rng.normal(
        loc=0.0, scale=(feature_std.values * NOISE_STD_FRAC),
        size=(n_affected, len(SHAPE_FEATURES))
    )
    blended = (
        (1 - BLEND_ALPHA) * X_obf.loc[mask, SHAPE_FEATURES].values
        + BLEND_ALPHA * benign_mean.values
        + noise
    )
    X_obf.loc[mask, SHAPE_FEATURES] = blended
    return X_obf


print(f"\nObfuscating {sum(y_test.isin(DARKNET_CLASSES))} / {len(y_test)} test rows "
      f"(blend_alpha={BLEND_ALPHA}, noise_std_frac={NOISE_STD_FRAC})")

X_test_obf = obfuscate(X_test, y_test)
X_train_obf = obfuscate(X_train, y_train)

# ============================================================
# RANDOM FOREST under attack
# ============================================================
print("\n" + "=" * 60)
print("RANDOM FOREST baseline under obfuscation attack")
print("=" * 60)

rf_clean = joblib.load(f"{RESULTS_DIR}/rf_model.joblib")

rf_results = {}

# Scenario (a): obfuscate TEST only (model trained clean, evaluated on obfuscated test)
y_pred = rf_clean.predict(X_test_obf)
rf_results["test_only"] = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
}
print(f"(a) Test-only obfuscation  -> acc={rf_results['test_only']['accuracy']:.4f}, "
      f"macro-F1={rf_results['test_only']['f1_macro']:.4f}")

# Scenario (b): obfuscate TRAIN only, evaluate on clean test
rf_train_obf = RandomForestClassifier(
    n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
)
rf_train_obf.fit(X_train_obf, y_train)
y_pred = rf_train_obf.predict(X_test)
rf_results["train_only"] = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
}
print(f"(b) Train-only obfuscation -> acc={rf_results['train_only']['accuracy']:.4f}, "
      f"macro-F1={rf_results['train_only']['f1_macro']:.4f}")

# Scenario (c): obfuscate BOTH (defender retrains on obfuscated data)
rf_both_obf = RandomForestClassifier(
    n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
)
rf_both_obf.fit(X_train_obf, y_train)
y_pred = rf_both_obf.predict(X_test_obf)
rf_results["both"] = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
}
print(f"(c) Both obfuscated (retrained) -> acc={rf_results['both']['accuracy']:.4f}, "
      f"macro-F1={rf_results['both']['f1_macro']:.4f}")

# ============================================================
# GRAPH MODEL under attack
# ============================================================
print("\n" + "=" * 60)
print("GRAPH-AUGMENTED model under obfuscation attack")
print("=" * 60)

graph_clf = joblib.load(f"{RESULTS_DIR}/graph_model.joblib")
graph_scaler = joblib.load(f"{RESULTS_DIR}/graph_scaler.joblib")
node_mean_features = joblib.load(f"{RESULTS_DIR}/node_mean_features.joblib")


def build_graph_features(X, ip_df):
    X_values = X[feature_cols].values
    src_agg = np.vstack([node_mean_features.get(ip, np.zeros(len(feature_cols)))
                          for ip in ip_df["Src IP"]])
    dst_agg = np.vstack([node_mean_features.get(ip, np.zeros(len(feature_cols)))
                          for ip in ip_df["Dst IP"]])
    return np.hstack([X_values, src_agg, dst_agg])


graph_results = {}

# (a) Test-only
Xg_test_obf = build_graph_features(X_test_obf, ip_test)
Xg_test_obf_scaled = graph_scaler.transform(Xg_test_obf)
y_pred = graph_clf.predict(Xg_test_obf_scaled)
graph_results["test_only"] = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
}
print(f"(a) Test-only obfuscation  -> acc={graph_results['test_only']['accuracy']:.4f}, "
      f"macro-F1={graph_results['test_only']['f1_macro']:.4f}")

# (b) Train-only: retrain graph model on obfuscated train, eval on clean test
Xg_train_obf = build_graph_features(X_train_obf, ip_train)
Xg_test_clean = build_graph_features(X_test, ip_test)
scaler_b = StandardScaler()
Xg_train_obf_scaled = scaler_b.fit_transform(Xg_train_obf)
Xg_test_clean_scaled = scaler_b.transform(Xg_test_clean)

graph_clf_b = MLPClassifier(
    hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
    alpha=1e-4, max_iter=100, random_state=RANDOM_STATE,
    early_stopping=True, n_iter_no_change=10,
)
graph_clf_b.fit(Xg_train_obf_scaled, y_train)
y_pred = graph_clf_b.predict(Xg_test_clean_scaled)
graph_results["train_only"] = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
}
print(f"(b) Train-only obfuscation -> acc={graph_results['train_only']['accuracy']:.4f}, "
      f"macro-F1={graph_results['train_only']['f1_macro']:.4f}")

# (c) Both: retrain on obfuscated train, eval on obfuscated test
scaler_c = StandardScaler()
Xg_train_obf_scaled_c = scaler_c.fit_transform(Xg_train_obf)
Xg_test_obf_scaled_c = scaler_c.transform(Xg_test_obf)

graph_clf_c = MLPClassifier(
    hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
    alpha=1e-4, max_iter=100, random_state=RANDOM_STATE,
    early_stopping=True, n_iter_no_change=10,
)
graph_clf_c.fit(Xg_train_obf_scaled_c, y_train)
y_pred = graph_clf_c.predict(Xg_test_obf_scaled_c)
graph_results["both"] = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
}
print(f"(c) Both obfuscated (retrained) -> acc={graph_results['both']['accuracy']:.4f}, "
      f"macro-F1={graph_results['both']['f1_macro']:.4f}")

# ============================================================
# Save combined results
# ============================================================
rf_clean_results = json.load(open(f"{RESULTS_DIR}/rf_clean_results.json"))
graph_clean_results = json.load(open(f"{RESULTS_DIR}/graph_clean_results.json"))

summary = {
    "attack_config": {
        "blend_alpha": BLEND_ALPHA,
        "noise_std_frac": NOISE_STD_FRAC,
        "shape_features": SHAPE_FEATURES,
        "darknet_classes_obfuscated": DARKNET_CLASSES,
        "benign_reference_class": BENIGN_CLASS,
    },
    "random_forest": {
        "clean": {
            "accuracy": rf_clean_results["clean_accuracy"],
            "f1_macro": rf_clean_results["clean_f1_macro"],
        },
        **rf_results,
    },
    "graph_model": {
        "clean": {
            "accuracy": graph_clean_results["clean_accuracy"],
            "f1_macro": graph_clean_results["clean_f1_macro"],
        },
        **graph_results,
    },
}

with open(f"{RESULTS_DIR}/attack_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Scenario':<20} {'RF Acc':>10} {'RF F1':>10} {'Graph Acc':>12} {'Graph F1':>10}")
for scenario in ["clean", "test_only", "train_only", "both"]:
    rf_r = summary["random_forest"][scenario]
    gr_r = summary["graph_model"][scenario]
    print(f"{scenario:<20} {rf_r['accuracy']:>10.4f} {rf_r['f1_macro']:>10.4f} "
          f"{gr_r['accuracy']:>12.4f} {gr_r['f1_macro']:>10.4f}")

print(f"\nSaved to {RESULTS_DIR}/attack_results.json")
