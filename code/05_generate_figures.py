

##Generates the figures and tables used in Results & Evaluation.

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project folder (one level above code/)
RESULTS_DIR = os.path.join(BASE, "results")
FIG_DIR = os.path.join(BASE, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

attack = json.load(open(f"{RESULTS_DIR}/attack_results.json"))
rf_clean = json.load(open(f"{RESULTS_DIR}/rf_clean_results.json"))
graph_clean = json.load(open(f"{RESULTS_DIR}/graph_clean_results.json"))

scenarios = ["clean", "test_only", "train_only", "both"]
scenario_labels = ["Clean", "Test-only\nattack", "Train-only\nattack", "Both\n(retrained)"]

rf_acc = [attack["random_forest"][s]["accuracy"] for s in scenarios]
rf_f1 = [attack["random_forest"][s]["f1_macro"] for s in scenarios]
gr_acc = [attack["graph_model"][s]["accuracy"] for s in scenarios]
gr_f1 = [attack["graph_model"][s]["f1_macro"] for s in scenarios]

# ---- Figure 1: Macro-F1 comparison bar chart ----
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(scenarios))
width = 0.35
ax.bar(x - width/2, rf_f1, width, label="Random Forest", color="#4C72B0")
ax.bar(x + width/2, gr_f1, width, label="Graph-Augmented Model", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(scenario_labels)
ax.set_ylabel("Macro F1-Score")
ax.set_ylim(0, 1.05)
ax.set_title("Macro F1-Score Under Obfuscation Attack")
ax.legend()
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(rf_f1):
    ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate(gr_f1):
    ax.text(i + width/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_f1_comparison.png", dpi=150)
print("Saved fig1_f1_comparison.png")

# ---- Figure 2: Accuracy comparison bar chart ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, rf_acc, width, label="Random Forest", color="#4C72B0")
ax.bar(x + width/2, gr_acc, width, label="Graph-Augmented Model", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(scenario_labels)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Accuracy Under Obfuscation Attack")
ax.legend()
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(rf_acc):
    ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate(gr_acc):
    ax.text(i + width/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_accuracy_comparison.png", dpi=150)
print("Saved fig2_accuracy_comparison.png")

# ---- Figure 3: Confusion matrices, clean vs test-only-attack, for RF ----
rf_cm_clean = np.array(rf_clean["confusion_matrix"])
classes = rf_clean["classes"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, cm, title in zip(
    axes,
    [rf_cm_clean],
    ["Random Forest -- Clean Data"],
):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
axes[1].axis("off")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_rf_confusion_clean.png", dpi=150)
print("Saved fig3_rf_confusion_clean.png")

# ---- Table: Results summary CSV ----
rows = []
for model_name, model_key in [("Random Forest", "random_forest"), ("Graph-Augmented Model", "graph_model")]:
    for s in scenarios:
        rows.append({
            "Model": model_name,
            "Scenario": s,
            "Accuracy": attack[model_key][s]["accuracy"],
            "Macro F1": attack[model_key][s]["f1_macro"],
        })
summary_df = pd.DataFrame(rows)
summary_df.to_csv(f"{RESULTS_DIR}/summary_table.csv", index=False)
print("\nSummary table:")
print(summary_df.to_string(index=False))

# F1 drop calculations for the write-up
print("\n--- Degradation analysis (test-only attack, the realistic scenario) ---")
rf_drop_f1 = rf_f1[0] - rf_f1[1]
gr_drop_f1 = gr_f1[0] - gr_f1[1]
print(f"RF macro-F1 drop:    {rf_drop_f1:.4f} ({rf_drop_f1/rf_f1[0]*100:.1f}% relative drop)")
print(f"Graph macro-F1 drop: {gr_drop_f1:.4f} ({gr_drop_f1/gr_f1[0]*100:.1f}% relative drop)")
print(f"Graph model retains {gr_f1[1]/gr_f1[0]*100:.1f}% of its clean F1 under attack")
print(f"RF retains {rf_f1[1]/rf_f1[0]*100:.1f}% of its clean F1 under attack")
