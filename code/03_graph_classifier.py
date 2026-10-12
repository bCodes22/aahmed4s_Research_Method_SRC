
#Simplified graph-based classifier informed by DGNN (Zhu et al., 2023/24).


import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
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
ip_train = pd.read_csv(f"{DATA_DIR}/ip_train.csv")
ip_test = pd.read_csv(f"{DATA_DIR}/ip_test.csv")

feature_cols = list(X_train.columns)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# --- Combine train+test for graph STRUCTURE only (transductive setting) ---
X_all = pd.concat([X_train, X_test], ignore_index=True)
ip_all = pd.concat([ip_train, ip_test], ignore_index=True)
n_train = len(X_train)

print("\nBuilding client-server graph from Src IP / Dst IP...")
G = nx.Graph()
for idx, row in ip_all.iterrows():
    src, dst = row["Src IP"], row["Dst IP"]
    G.add_edge(src, dst)
    # tag each edge occurrence with the flow row index for later lookup
    if "flow_idx" not in G[src][dst]:
        G[src][dst]["flow_idx"] = []
    G[src][dst]["flow_idx"].append(idx)

print(f"Graph: {G.number_of_nodes()} nodes (IPs), {G.number_of_edges()} edges (unique IP pairs)")

# --- Precompute mean flow-feature vector per node (GraphSAGE-style mean aggregator) ---
print("Computing per-node neighbor feature aggregation...")
node_flow_indices = {node: [] for node in G.nodes()}
for idx, row in ip_all.iterrows():
    node_flow_indices[row["Src IP"]].append(idx)
    node_flow_indices[row["Dst IP"]].append(idx)

X_values = X_all[feature_cols].values
node_mean_features = {}
for node, idxs in node_flow_indices.items():
    node_mean_features[node] = X_values[idxs].mean(axis=0)

# --- Build graph-augmented feature matrix: [own features | src-node agg | dst-node agg] ---
print("Assembling graph-augmented feature matrix...")
src_agg = np.vstack([node_mean_features[ip] for ip in ip_all["Src IP"]])
dst_agg = np.vstack([node_mean_features[ip] for ip in ip_all["Dst IP"]])
X_graph_all = np.hstack([X_values, src_agg, dst_agg])

X_graph_train = X_graph_all[:n_train]
X_graph_test = X_graph_all[n_train:]

print(f"Graph-augmented feature dim: {X_graph_all.shape[1]} "
      f"(= {len(feature_cols)} own + {len(feature_cols)} src-agg + {len(feature_cols)} dst-agg)")

# --- Scale features (required for MLP convergence) ---
scaler = StandardScaler()
X_graph_train_scaled = scaler.fit_transform(X_graph_train)
X_graph_test_scaled = scaler.transform(X_graph_test)

# --- Train MLP classifier head on graph-augmented representation ---
print("\nTraining graph-augmented MLP classifier...")
clf = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    max_iter=100,
    random_state=RANDOM_STATE,
    early_stopping=True,
    n_iter_no_change=10,
)
clf.fit(X_graph_train_scaled, y_train)

y_pred = clf.predict(X_graph_test_scaled)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print(f"\nClean-data accuracy: {acc:.4f}")
print(f"Clean-data macro F1: {f1_macro:.4f}")
print(f"Clean-data weighted F1: {f1_weighted:.4f}")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
print("\nConfusion matrix (rows=true, cols=pred):")
print("Classes:", list(clf.classes_))
print(cm)

# --- Save model, scaler, graph artifacts ---
joblib.dump(clf, f"{RESULTS_DIR}/graph_model.joblib")
joblib.dump(scaler, f"{RESULTS_DIR}/graph_scaler.joblib")
joblib.dump(node_mean_features, f"{RESULTS_DIR}/node_mean_features.joblib")

results = {
    "model": "Graph-augmented MLP (GraphSAGE-style mean aggregation over client/server graph + MLP head)",
    "graph_nodes": G.number_of_nodes(),
    "graph_edges": G.number_of_edges(),
    "feature_dim": X_graph_all.shape[1],
    "clean_accuracy": float(acc),
    "clean_f1_macro": float(f1_macro),
    "clean_f1_weighted": float(f1_weighted),
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "classes": list(clf.classes_),
}
with open(f"{RESULTS_DIR}/graph_clean_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved model and results to {RESULTS_DIR}")
