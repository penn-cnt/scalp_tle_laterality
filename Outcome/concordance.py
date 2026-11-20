# Fig 4B / 4D: concordant laterality probability vs surgical outcome (Engel, ILAE).
# Same analysis as pickle/machine_learning/surgical_outcome_2_year_Jan0726.ipynb (cells 1-11):
#   - outcome: juri_surgery_outcomes_extracted_DEID.csv (latest of 6/12/24/36-month) + REDCap (deid_with_redcap_Dec1025.csv)
#   - test set: test_104_no_seizures.csv
# Expected: Engel p = 0.0365, ILAE p = 0.0841
# The GitHub version (Dataset/outcome.csv) is kept in concordance_github.py.

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "..", "Dataset", "training_261.csv")
test_path = os.path.join(script_dir, "..", "Dataset", "test_104_no_seizures.csv")
# clinical outcome files are not part of the public Dataset; set CLINICAL_DIR to their folder
clinical_dir = os.environ.get("CLINICAL_DIR", "/mnt/sauce/littlab/users/jurikim/ied_yesno/pickle/machine_learning")
deid_path = os.path.join(clinical_dir, "deid_with_redcap_Dec1025.csv")
outcome_path = os.path.join(clinical_dir, "juri_surgery_outcomes_extracted_DEID.csv")
save_path_4B = os.path.join(script_dir, "..", "Visualization", "Fig4B.jpg")
save_path_4D = os.path.join(script_dir, "..", "Visualization", "Fig4D.jpg")

# ------------------------------------------------------------
# 2. Surgical outcome table (REDCap TLE + extracted outcomes)
# ------------------------------------------------------------
dei_df = pd.read_csv(deid_path)
dei_df_TLE = dei_df[dei_df['single_loc'] == 'temporal'].copy()

int_str = dei_df_TLE["Intervention_cnt_redcap"].astype(str)
pec_str = dei_df_TLE["pec_intervention"].astype(str)
has_laser_or_resection_int = int_str.str.contains("Laser Ablation|Resection", regex=True, na=False)
has_laser_or_resection_pec = pec_str.str.contains("Laser Ablation|Resection", regex=True, na=False)
has_other_int = int_str.str.contains("Other (specify)", na=False)
has_other_pec = pec_str.str.contains("Other (specify)", na=False)

# Laser/Resection on one side and Other (specify) on the other: always keep
keep_laser_resect_other = (
    (has_laser_or_resection_int & has_other_pec) |
    (has_laser_or_resection_pec & has_other_int)
)
mask_laser_resect = (
    ((dei_df_TLE['Intervention_cnt_redcap'] == 'Laser Ablation') & (dei_df_TLE['pec_intervention'] == 'Resection')) |
    ((dei_df_TLE['Intervention_cnt_redcap'] == 'Resection') & (dei_df_TLE['pec_intervention'] == 'Laser Ablation'))
)
mask_dbs_rns_vns = (
    int_str.str.contains('DBS|RNS|VNS', regex=True, na=False) |
    pec_str.str.contains('DBS|RNS|VNS', regex=True, na=False)
)
mask_no_int_nan = (
    ((dei_df_TLE['Intervention_cnt_redcap'] == 'No intervention received') & dei_df_TLE['pec_intervention'].isna()) |
    ((dei_df_TLE['pec_intervention'] == 'No intervention received') & dei_df_TLE['Intervention_cnt_redcap'].isna())
)
mask_both_nan = dei_df_TLE['Intervention_cnt_redcap'].isna() & dei_df_TLE['pec_intervention'].isna()
mask_engel_ilae_nan = dei_df_TLE['engel_class_pecclinical'].isna() | dei_df_TLE['ilae_category_pecclinical'].isna()

mask_drop_rows = (
    mask_laser_resect | mask_dbs_rns_vns | mask_no_int_nan | mask_both_nan | mask_engel_ilae_nan
) & (~keep_laser_resect_other)

dei_df_TLE_new = (
    dei_df_TLE[~mask_drop_rows]
    .sort_values(by=['patient_id'])
    .drop_duplicates(subset='patient_id', keep='first')
    .reset_index(drop=True)
)

# Extracted surgery outcomes
outcome_df = pd.read_csv(outcome_path)
outcome_df = outcome_df[outcome_df["type_surgery"].notna()].copy()
outcome_df["type_surgery"] = outcome_df["type_surgery"].replace({
    "laser ablation": "Laser Ablation",
    "resection": "Resection"
})
outcome_df = outcome_df.rename(columns={
    "engel_12months": "engel_class_pecclinical",
    "engel_24months": "engel_class_2_pecclinical",
    "engel_post36months": "engel_class_3_pecclinical",
    "ilae_12months": "ilae_category_pecclinical",
    "ilae_24months": "ilae_category_2_pecclinical",
    "ilae_post36months": "ilae_category_3_pecclinical",
})
for col in ["ilae_6months", "engel_6months"]:
    if col not in outcome_df.columns:
        outcome_df[col] = pd.NA

ilae_cols = ["ilae_6months", "ilae_category_pecclinical", "ilae_category_2_pecclinical", "ilae_category_3_pecclinical"]
engel_cols = ["engel_6months", "engel_class_pecclinical", "engel_class_2_pecclinical", "engel_class_3_pecclinical"]


def extract_ilae_code(s):
    # e.g. "... (past 12 months) 1a = ..." -> "1a"
    if pd.isna(s):
        return pd.NA
    s = str(s)
    m = re.search(r"\(past 12 months\)\s*([^\s=]+)", s)
    return m.group(1) if m else s


for col in ilae_cols:
    outcome_df[col] = outcome_df[col].apply(extract_ilae_code)
for col in engel_cols:
    outcome_df[col] = (
        outcome_df[col].astype(str).str.split(":").str[0].str.strip()
        .where(outcome_df[col].notna(), pd.NA)
    )

dei_df_TLE_outcome = pd.merge(dei_df_TLE_new, outcome_df, on="patient_id", how="outer", suffixes=("", "_outcome"))

# fill missing REDCap values with the extracted outcome values
for col in engel_cols + ilae_cols:
    col_out = col + "_outcome"
    if col_out in dei_df_TLE_outcome.columns:
        dei_df_TLE_outcome[col] = dei_df_TLE_outcome[col].where(
            ~dei_df_TLE_outcome[col].isna(), dei_df_TLE_outcome[col_out])
        dei_df_TLE_outcome = dei_df_TLE_outcome.drop(columns=[col_out])

mask_missing_int = dei_df_TLE_outcome["Intervention_cnt_redcap"].isna()
dei_df_TLE_outcome.loc[mask_missing_int, "Intervention_cnt_redcap"] = dei_df_TLE_outcome.loc[mask_missing_int, "type_surgery"]

# 1014 and 985 have no outcome; 930 is bilateral
dei_df_TLE_outcome = dei_df_TLE_outcome[~dei_df_TLE_outcome["patient_id"].isin([1014, 985, 930])]

# Final outcome = latest available follow-up (36 > 24 > 12 > 6 months)
ilae_priority = ["ilae_category_3_pecclinical", "ilae_category_2_pecclinical", "ilae_category_pecclinical", "ilae_6months"]
engel_priority = ["engel_class_3_pecclinical", "engel_class_2_pecclinical", "engel_class_pecclinical", "engel_6months"]
dei_df_TLE_outcome["ILAE_final"] = pd.NA
for col in ilae_priority:
    dei_df_TLE_outcome["ILAE_final"] = dei_df_TLE_outcome["ILAE_final"].where(
        dei_df_TLE_outcome["ILAE_final"].notna(), dei_df_TLE_outcome[col])
dei_df_TLE_outcome["Engel_final"] = pd.NA
for col in engel_priority:
    dei_df_TLE_outcome["Engel_final"] = dei_df_TLE_outcome["Engel_final"].where(
        dei_df_TLE_outcome["Engel_final"].notna(), dei_df_TLE_outcome[col])

# two more patients shared by Erin on Dec 19, 2025
new_rows = pd.DataFrame([
    {"patient_id": 182, "Intervention_cnt_redcap": "Resection", "ILAE_final": 1, "Engel_final": "IA"},
    {"patient_id": 720, "Intervention_cnt_redcap": "Resection", "ILAE_final": 1, "Engel_final": "IA"},
]).reindex(columns=dei_df_TLE_outcome.columns)
dei_df_TLE_outcome = pd.concat([dei_df_TLE_outcome, new_rows], ignore_index=True)

# 699, 965 are not in the patients used for model training
dei_df_TLE_outcome = dei_df_TLE_outcome[~dei_df_TLE_outcome['patient_id'].isin([699, 965])]

# ------------------------------------------------------------
# 3. Train left-vs-rest / right-vs-rest models (SAI, threshold 0.43)
# ------------------------------------------------------------
THRESHOLD = 0.43
features = ['SAI']

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_train = df_train.drop_duplicates(subset='patient_id', keep='first')
df_train = df_train[
    (df_train['spike_rate_min'] > 0.016) &
    (~df_train['SAI'].isna()) &
    (~df_train['single_lat'].isna())
]
df_test = df_test.drop_duplicates(subset='patient_id', keep='first')

df_train_sub = df_train[df_train['threshold'] == THRESHOLD]
models = {}
for side in ['left', 'right']:
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, class_weight='balanced')
    model.fit(df_train_sub[features], (df_train_sub['single_lat'].str.lower() == side).astype(int))
    models[side] = model

# ------------------------------------------------------------
# 4. Concordance probability for operated patients (training + test)
# ------------------------------------------------------------
df_all_outcome = pd.concat([
    df_train.merge(dei_df_TLE_outcome, on="patient_id", how="left"),
    df_test.merge(dei_df_TLE_outcome, on="patient_id", how="left"),
], ignore_index=True).sort_values(["patient_id"]).reset_index(drop=True)


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


df_all_outcome["engel_binary"] = df_all_outcome["Engel_final"].map(engel_binary)
df_all_outcome["ilae_binary"] = df_all_outcome["ILAE_final"].map(ilae_binary)
df_all_outcome = df_all_outcome.dropna(subset=["engel_binary", "ilae_binary"], how="all")

df_sub_all = df_all_outcome[df_all_outcome["threshold"] == THRESHOLD].copy()
df_sub_all = df_sub_all[df_sub_all["Engel_final"].notna()].sort_values(["patient_id"]).reset_index(drop=True)
df_sub_all["surgery_side"] = df_sub_all["single_lat_x"].str.lower()

# probability that the model agrees with the operated side
df_sub_all["concordance_prob"] = np.nan
for side, model in models.items():
    mask = df_sub_all["surgery_side"] == side
    if mask.any():
        pos_idx = list(model.classes_).index(1)
        df_sub_all.loc[mask, "concordance_prob"] = model.predict_proba(df_sub_all.loc[mask, features])[:, pos_idx]

df_all = df_sub_all[df_sub_all["surgery_side"].isin(["left", "right"])].copy()

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


plot_mannwhitney(df_all, "engel_binary", "Engel I", "Engel II+", "Concordance vs Engel", save_path_4B)
plot_mannwhitney(df_all, "ilae_binary", "ILAE 1–2", "ILAE 3+", "Concordance vs ILAE", save_path_4D)
