# Fig 4B / 4D: concordant laterality probability vs surgical outcome (Engel, ILAE).
# Dataset/outcome.csv: operated patients (left or right surgery) with their latest available
# Engel and ILAE outcome (36, 24, 12 or 6 months).

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu

# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "..", "Dataset", "training_261.csv")
test_path = os.path.join(script_dir, "..", "Dataset", "test_104_no_seizures.csv")
outcome_path = os.path.join(script_dir, "..", "Dataset", "outcome.csv")
save_path_4B = os.path.join(script_dir, "..", "Visualization", "Fig4B.jpg")
save_path_4D = os.path.join(script_dir, "..", "Visualization", "Fig4D.jpg")

THRESHOLD = 0.43
features = ['SAI']

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_outcome = pd.read_csv(outcome_path, dtype={'engel_stage': str, 'ilae_stage': str})
# side of surgery = lateralization label of the operated patient
df_outcome = df_outcome.rename(columns={'single_lat': 'surgery_side'})[
    ['patient_id', 'surgery_side', 'engel_stage', 'ilae_stage']]

df_train = df_train.drop_duplicates(subset='patient_id', keep='first')
df_train = df_train[
    (df_train['spike_rate_min'] > 0.016) &
    (~df_train['SAI'].isna()) &
    (~df_train['single_lat'].isna())
]
df_test = df_test.drop_duplicates(subset='patient_id', keep='first')

# ------------------------------------------------------------
# 2. Train left-vs-rest / right-vs-rest models (SAI, threshold 0.43)
# ------------------------------------------------------------
df_train_sub = df_train[df_train['threshold'] == THRESHOLD]
models = {}
for side in ['left', 'right']:
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, class_weight='balanced')
    model.fit(df_train_sub[features], (df_train_sub['single_lat'].str.lower() == side).astype(int))
    models[side] = model

# ------------------------------------------------------------
# 3. Binary outcomes
# ------------------------------------------------------------
def engel_binary(v):
    # Engel I (IA-ID) = good (1), II-IV = poor (0)
    if pd.isna(v):
        return np.nan
    return 1 if str(v).strip().upper() in ["IA", "IB", "IC", "ID"] else 0


def ilae_binary(v):
    # ILAE 1, 1a, 2 = good (1), 3-6 = poor (0)
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in ["1", "1a"]:
        return 1
    try:
        x = float(s)
    except Exception:
        return np.nan
    if 1 <= x <= 2:
        return 1
    if 3 <= x <= 6:
        return 0
    return np.nan


df_outcome["engel_binary"] = df_outcome["engel_stage"].map(engel_binary)
df_outcome["ilae_binary"] = df_outcome["ilae_stage"].map(ilae_binary)

# ------------------------------------------------------------
# 4. Concordance probability for operated patients (training + test)
# ------------------------------------------------------------
df_all = pd.concat([df_train, df_test], ignore_index=True)
df_all = df_all[df_all["threshold"] == THRESHOLD]
df_all = df_all.merge(df_outcome, on="patient_id", how="inner").sort_values("patient_id").reset_index(drop=True)

# probability that the model agrees with the operated side
df_all["concordance_prob"] = np.nan
for side, model in models.items():
    mask = df_all["surgery_side"] == side
    if mask.any():
        pos_idx = list(model.classes_).index(1)
        df_all.loc[mask, "concordance_prob"] = model.predict_proba(df_all.loc[mask, features])[:, pos_idx]

# ------------------------------------------------------------
# 5. Mann-Whitney U and jitter plots
# ------------------------------------------------------------
rng = np.random.default_rng(0)  # fixed jitter so the figure is reproducible


def plot_mannwhitney(df, bin_col, good_label, poor_label, title, save_path):
    d = df[~df[bin_col].isna()].dropna(subset=["concordance_prob"]).copy()
    if d.empty or d[bin_col].nunique() < 2:
        print(f"{title}: insufficient data")
        return

    g1 = d[d[bin_col] == 1]["concordance_prob"]
    g0 = d[d[bin_col] == 0]["concordance_prob"]
    stat, p = mannwhitneyu(g1, g0, alternative="two-sided")
    print(f"{title}: Mann-Whitney U = {stat:.3f}, p = {p:.4g} (n good = {len(g1)}, n poor = {len(g0)})")

    d["x"] = d[bin_col].map({1: 0, 0: 1}).astype(float)
    d["x_jitter"] = d["x"] + rng.uniform(-0.1, 0.1, size=len(d))
    d["dot_color"] = d["surgery_side"].map({"left": "blue", "right": "red"}).fillna("gray")

    plt.figure(figsize=(6, 5))
    plt.scatter(d["x_jitter"], d["concordance_prob"], c=d["dot_color"], edgecolors="black", alpha=0.8, s=80)
    plt.xticks([0, 1], [good_label, poor_label])
    plt.xlim(-0.5, 1.5)
    plt.ylim(0, 1.1)
    plt.ylabel("Concordant laterality probability")
    plt.title(title)
    plt.text(0.5, 1.02, f"Mann-Whitney p = {p:.3f}", ha="center", fontsize=11)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


print(f"Operated patients: {len(df_all)}")
plot_mannwhitney(df_all, "engel_binary", "Engel I", "Engel II+", "Concordance vs Engel", save_path_4B)
plot_mannwhitney(df_all, "ilae_binary", "ILAE 1–2", "ILAE 3+", "Concordance vs ILAE", save_path_4D)
