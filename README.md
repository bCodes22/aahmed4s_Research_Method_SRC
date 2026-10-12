# Detecting Malicious Darknet Traffic with Machine Learning

Code and reproducibility package for CY530 Network Security, Assignment 2.
Author: Ajwad Ahmed

## What this project does

This project tests whether a more sophisticated classifier is also more robust against an attacker. It trains two models on the CIC-Darknet2020 dataset and compares them on traffic-type classification (Tor, Non-Tor, VPN, Non-VPN):

1. **Random Forest baseline**, replicating Rust-Nguyen et al. (2023).
2. **Graph-augmented classifier**, a simplified stand-in for the DGNN of Zhu et al. (2024). Each flow gets its own 76 features plus the mean features of its source IP and destination IP (228 features total), then goes into an MLP.

Both models are then attacked with a feature-space obfuscation attack in three scenarios: test data only, training data only, and both.

## Results

| Scenario | RF accuracy | RF macro F1 | Graph accuracy | Graph macro F1 |
|---|---|---|---|---|
| Clean data | 0.9805 | 0.9522 | 0.9980 | 0.9810 |
| Attack on test data only | 0.8269 | 0.4152 | 0.9972 | 0.9793 |
| Attack on training data only | 0.6595 | 0.1987 | 0.9910 | 0.9679 |
| Attack on both | 0.9607 | 0.9211 | 0.9982 | 0.9781 |

Under the test-only attack the Random Forest loses 56.4% of its macro F1. The graph model loses 0.2%. The full discussion, including limitations, is in the paper.

## Repository layout

```
code/
  01_prepare_data.py                 Cleans the raw CSV, selects features, makes the train/test split
  02_random_forest_baseline.py       Trains and evaluates the Random Forest
  03_graph_classifier.py             Builds the IP graph and trains the graph-augmented MLP
  04_obfuscation_attack.py           Runs the three attack scenarios on both models
  05_generate_figures.py             Makes the figures and the summary table
  06_per_class_attack_breakdown.py   Per-class precision/recall/F1 under the test-only attack
results/
  rf_clean_results.json              Random Forest clean-data metrics
  graph_clean_results.json           Graph model clean-data metrics
  attack_results.json                Results for all three attack scenarios
  summary_table.csv                  Flat version of the results table
figures/
  fig1_f1_comparison.png             Macro F1 across scenarios
  fig2_accuracy_comparison.png       Accuracy across scenarios
  fig3_rf_confusion_clean.png        Random Forest confusion matrix, clean data
screenshots/                         Terminal output from my own run of each script
requirements.txt                     Python dependencies
README.md                            This file
```

The dataset and the large model files are not in the repository (see `.gitignore`). The Random Forest model is about 100 MB and the dataset is about 600 MB. Everything can be regenerated with the steps below.

## How to run it

**1. Get the dataset.** Download CIC-Darknet2020 from the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/darknet2020.html

**2. Set up the folder.** Put the file next to the code like this:

```
project/
  Darknet.CSV
  code/
  requirements.txt
```

If the CSV is somewhere else, set the `DARKNET_CSV` environment variable to its full path instead. The scripts create `data/`, `results/` and `figures/` themselves, so no paths need to be edited.

**3. Install the packages** (Python 3.10 or newer), from `project/`:

```
pip install -r requirements.txt
```

**4. Run the scripts in order**, from inside `project/code/`:

```
python 01_prepare_data.py
python 02_random_forest_baseline.py
python 03_graph_classifier.py
python 04_obfuscation_attack.py
python 05_generate_figures.py
python 06_per_class_attack_breakdown.py
```

On Windows, use `py` instead of `python` if `python` is not recognized. Command Prompt works if PowerShell blocks scripts.

No GPU is needed. The whole pipeline takes roughly 5 to 10 minutes, mostly Random Forest training on about 113,000 rows.

## Reproducibility

All randomness is seeded with `RANDOM_STATE = 42`. Running the scripts on the same raw CSV should give the numbers in the table above. Tiny differences are possible because scikit-learn trains the Random Forest in parallel (`n_jobs=-1`).

## Key settings

- **Data:** 141,530 raw rows, 141,481 after removing 49 rows with non-numeric or infinite values. Stratified 80/20 split: 113,184 train, 28,297 test.
- **Random Forest:** 200 trees, `class_weight="balanced"`, everything else at scikit-learn defaults.
- **Graph model:** IPs are nodes and flows are edges. The graph is built over train and test together (transductive), but labels come only from the training set. The classifier is `MLPClassifier(hidden_layer_sizes=(128, 64))`.
- **Attack:** 43 "shape" features (packet length, inter-arrival time, duration, bytes/s, packets/s, active/idle timing) are blended 60% toward the Non-Tor mean, with Gaussian noise at 5% of each feature's standard deviation. It is applied to Tor, VPN and Non-VPN rows. Structural features such as TCP flag counts are left alone.

## Limitations

- The graph model uses mean aggregation, not the attention mechanism in the original DGNN. It tests the idea that relationships between hosts help, but it is not a full reimplementation.
- The graph model relies heavily on per-IP averages, which may act as a fingerprint of particular machines in this dataset. It might not hold up on traffic from different networks.
- The attack changes flow features directly. It does not rebuild real packets, so it is an approximation of what an attacker could do.
- Pre-generated CICFlowMeter features were used. CICFlowMeter was not re-run on raw captures.

## Use of AI assistance

I used Claude as an assistant on this project, as permitted by the assignment.
Working from my direction and the Assignment 1 plan, Claude helped generate
the Python graphs for the experiments and help me understand the graphs/outputs
of my code. It also helped me create the tables on my README file for a clear understanding. I ran all of the code on my own machine, checked the results against
the report, and reviewed and edited the writing

## References

- N. Rust-Nguyen, S. Sharma and M. Stamp, "Darknet traffic classification and adversarial attacks using machine learning," *Computers & Security*, vol. 127, 2023. https://doi.org/10.1016/j.cose.2023.103098
- Y. Zhu et al., "DGNN: Accurate darknet application classification adopting attention graph neural network," *IEEE TNSM*, vol. 21, no. 2, 2024. https://doi.org/10.1109/TNSM.2023.3344580
- Canadian Institute for Cybersecurity, CIC-Darknet2020. https://www.unb.ca/cic/datasets/darknet2020.html
