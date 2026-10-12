
#Per-class precision / recall / F1 for both models under the realistic
#TEST-ONLY obfuscation attack.

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze("columns")
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze("columns")
ip_test = pd.read_csv(f"{DATA_DIR}/ip_test.csv")
feature_cols = list(X_train.columns)

# --- same attack settings as script 04 ---
SHAPE_FEATURES = [c for c in feature_cols if any(k in c for k in [
    "Packet Length", "IAT", "Flow Duration", "Bytes/s", "Packets/s",
    "Average Packet Size", "Segment Size", "Active", "Idle",
])]
DARKNET_CLASSES = ["Tor", "VPN", "NonVPN"]
BENIGN_CLASS = "Non-Tor"
BLEND_ALPHA = 0.6
NOISE_STD_FRAC = 0.05
benign_mean = X_train.loc[y_train == BENIGN_CLASS, SHAPE_FEATURES].mean()
feature_std = X_train[SHAPE_FEATURES].std().replace(0, 1e-6)


def obfuscate(X, y):
    X_obf = X.copy().astype(float)
    mask = y.isin(DARKNET_CLASSES)
    noise = rng.normal(0.0, feature_std.values * NOISE_STD_FRAC,
                       size=(mask.sum(), len(SHAPE_FEATURES)))
    X_obf.loc[mask, SHAPE_FEATURES] = ((1 - BLEND_ALPHA) * X_obf.loc[mask, SHAPE_FEATURES].values
                                      + BLEND_ALPHA * benign_mean.values + noise)
    return X_obf


X_test_obf = obfuscate(X_test, y_test)  # first draw from the seeded rng, as in script 04

# --- Random Forest ---
rf = joblib.load(f"{RESULTS_DIR}/rf_model.joblib")
print("=" * 60)
print("RANDOM FOREST, test-only attack, per class")
print("=" * 60)
print(classification_report(y_test, rf.predict(X_test_obf), digits=3, zero_division=0))

# --- Graph model ---
clf = joblib.load(f"{RESULTS_DIR}/graph_model.joblib")
scaler = joblib.load(f"{RESULTS_DIR}/graph_scaler.joblib")
node_mean = joblib.load(f"{RESULTS_DIR}/node_mean_features.joblib")
src = np.vstack([node_mean[ip] for ip in ip_test["Src IP"]])
dst = np.vstack([node_mean[ip] for ip in ip_test["Dst IP"]])
Xg = scaler.transform(np.hstack([X_test_obf[feature_cols].values, src, dst]))
print("=" * 60)
print("GRAPH MODEL, test-only attack, per class")
print("=" * 60)
print(classification_report(y_test, clf.predict(Xg), digits=3, zero_division=0))
