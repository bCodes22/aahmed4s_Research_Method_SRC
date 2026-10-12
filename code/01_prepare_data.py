##Loads CIC_Darknet data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project folder (one level above code/)
# Put the dataset at <project>/Darknet.CSV, or set the DARKNET_CSV environment variable.
RAW_PATH = os.environ.get("DARKNET_CSV", os.path.join(BASE, "Darknet.CSV"))
OUT_DIR = os.path.join(BASE, "data")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42

print("Loading raw CSV...")
df = pd.read_csv(RAW_PATH, low_memory=False, on_bad_lines="skip")
print(f"Raw shape: {df.shape}")

# --- Fix duplicate 'Label' columns (pandas auto-renames the 2nd 'Label' -> 'Label.1') ---
df = df.rename(columns={"Label": "traffic_type", "Label.1": "app_type"})

# --- Clean traffic_type (the label used for this study) ---
# Known values: Non-Tor, NonVPN, VPN, Tor -- already consistent in this dataset
df["traffic_type"] = df["traffic_type"].astype(str).str.strip()
print("\nTraffic-type class counts (raw):")
print(df["traffic_type"].value_counts())

# --- Clean app_type for completeness / future work (merge casing duplicates) ---
df["app_type"] = df["app_type"].astype(str).str.strip().str.upper()
app_map = {
    "AUDIO-STREAMING": "AUDIO-STREAMING",
    "VIDEO-STREAMING": "VIDEO-STREAMING",
    "FILE-TRANSFER": "FILE-TRANSFER",
}
df["app_type"] = df["app_type"].replace(app_map)
print("\nApp-type class counts (cleaned, for reference only):")
print(df["app_type"].value_counts())

# --- Select flow-level numeric features (exclude IP/port/flow-ID/timestamp identifiers) ---
drop_cols = [
    "Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol",
    "Timestamp", "traffic_type", "app_type",
]
feature_cols = [c for c in df.columns if c not in drop_cols]

# Coerce all feature columns to numeric, drop rows that fail / contain inf
X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce")
X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
valid_mask = X_raw.notna().all(axis=1)
print(f"\nDropping {(~valid_mask).sum()} rows with non-numeric/inf/NaN feature values "
      f"({(~valid_mask).sum() / len(df) * 100:.2f}% of data)")

df_clean = df.loc[valid_mask].reset_index(drop=True)
X = X_raw.loc[valid_mask].reset_index(drop=True)
y = df_clean["traffic_type"]

print(f"\nFinal clean shape: X={X.shape}, y={y.shape}")
print("\nFinal class distribution:")
print(y.value_counts())
print("\nClass distribution (%):")
print((y.value_counts(normalize=True) * 100).round(2))

# --- Keep IP columns alongside the split for graph construction later ---
ip_cols = df_clean[["Src IP", "Dst IP"]].copy()

# --- Stratified train/test split (80/20) ---
X_train, X_test, y_train, y_test, ip_train, ip_test = train_test_split(
    X, y, ip_cols, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
print("\nTrain class distribution:")
print(y_train.value_counts())
print("\nTest class distribution:")
print(y_test.value_counts())

# --- Save everything ---
X_train.to_csv(f"{OUT_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{OUT_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{OUT_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{OUT_DIR}/y_test.csv", index=False)
ip_train.to_csv(f"{OUT_DIR}/ip_train.csv", index=False)
ip_test.to_csv(f"{OUT_DIR}/ip_test.csv", index=False)

summary = {
    "raw_rows": int(len(df)),
    "clean_rows": int(len(df_clean)),
    "dropped_rows": int((~valid_mask).sum()),
    "n_features": len(feature_cols),
    "feature_cols": feature_cols,
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "class_distribution_full": y.value_counts().to_dict(),
    "class_distribution_train": y_train.value_counts().to_dict(),
    "class_distribution_test": y_test.value_counts().to_dict(),
    "random_state": RANDOM_STATE,
}
with open(f"{OUT_DIR}/prep_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved train/test splits and prep_summary.json to", OUT_DIR)
