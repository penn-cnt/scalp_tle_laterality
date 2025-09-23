import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import ttest_ind, mannwhitneyu

# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
df_train = os.path.join(script_dir, "/Dataset/training_261.csv")
df_test = os.path.join(script_dir, "/Dataset/test_104.csv")
cnt_surgical = os.path.join(script_dir, "/Dataset/outcome.csv")
save_path_4B = os.path.join(script_dir, "Visualization", "Fig4B.jpg")
save_path_4D = os.path.join(script_dir, "Visualization", "Fig4D.jpg")

df_train = pd.read_csv(df_train)
df_test = pd.read_csv(df_test)
cnt_surgical = pd.read_csv(cnt_surgical)
# ------------------------------------------------------------
# 2. Data Preprocessing and Merging
# ------------------------------------------------------------
# Clean and filter training data
df_train = df_train.drop_duplicates(subset='patient_id', keep='first')
df_train = df_train[
    (df_train['spike_rate_min'] > 0.016) &
    (~df_train['SAI'].isna()) &
    (~df_train['single_lat'].isna())
]

# Map Engel stage to binary outcome
df_engel = cnt_surgical.copy()
engel_I = ['I']
engel_II_plus = ['II', 'III', 'IV']
df_engel = df_engel[df_engel['engel_stage_cnt'].isin(engel_I + engel_II_plus)]
df_engel['engel_binary'] = df_engel['engel_stage_cnt'].apply(lambda x: 1 if x in engel_I else 0)

# Merge with Engel data
df_train_engel = df_train.merge(df_engel[['patient_id', 'engel_binary']], on='patient_id', how='left')
df_train_engel = df_train_engel[df_train_engel['engel_binary'].notna()].copy()
df_test_engel = df_test.merge(df_engel[['patient_id', 'engel_binary']], on='patient_id', how='left')
df_test_engel = df_test_engel[df_test_engel['engel_binary'].notna()].copy()
df_all_engel = pd.concat([df_train_engel, df_test_engel], ignore_index=True)

# Map descriptive ILAE labels to concise ILAE stage labels
ilae_label_map = {
    "Seizure free since surgery, no auras": "ILAE 1",
    "Auras only, no other seizures": "ILAE 2",
    "Rare seizures (1-3 seizure days per year)": "ILAE 3",
    "Seizure reduction >50% (but >3 seizure days/year)": "ILAE 4",
    "Seizure reduction >50% (but >3 seizure days/year": "ILAE 4",
    "No change (between 50% seizure reduction and 100% increase)": "ILAE 5"
}

def map_ilae_descriptive_to_short(x: str) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)): return np.nan
    s = str(x).strip()
    if s in ilae_label_map: return ilae_label_map[s]
    sl = s.lower()
    if "seizure free since surgery" in sl and "no auras" in sl: return "ILAE 1"
    if "auras only" in sl and "no other seizures" in sl: return "ILAE 2"
    if "rare seizures" in sl or "1-3 seizure days per year" in sl: return "ILAE 3"
    if "seizure reduction >50%" in sl: return "ILAE 4"
    if "no change" in sl and "50% seizure reduction" in sl: return "ILAE 5"
    return np.nan

cnt_surgical["ilae_stage_cnt"] = cnt_surgical["ilae_stage_cnt"].apply(map_ilae_descriptive_to_short)
ilae_good = {"ILAE 1", "ILAE 2"}
ilae_bad = {"ILAE 3", "ILAE 4", "ILAE 5"}
cnt_surgical["ilae_binary"] = cnt_surgical["ilae_stage_cnt"].apply(
    lambda x: 1 if x in ilae_good else (0 if x in ilae_bad else np.nan)
)

# Merge with ILAE data
df_all_ilae = df_all_engel.drop(columns=["engel_binary", "engel_stage_cnt"], errors="ignore").merge(
    cnt_surgical[["admission_id", "ilae_stage_cnt", "ilae_binary"]],
    on="admission_id",
    how="left"
)

# ------------------------------------------------------------
# 3. Concordance Analysis and Plotting
# ------------------------------------------------------------
# Define best model configs and feature sets
best_configs = [
    {
        "task": "left_vs_rest",
        "threshold": 0.43,
        "feature_set": "SAI",
        "model": LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, class_weight='balanced'),
        "label": "Left vs Right/Bilateral"
    },
    {
        "task": "right_vs_rest",
        "threshold": 0.43,
        "feature_set": "SAI",
        "model": LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, class_weight='balanced'),
        "label": "Right vs Left/Bilateral"
    }
]

feature_sets = {
    "SAI": ['SAI'],
}

# Run modeling and build concordance data
concordance_data_engel = []
concordance_data_ilae = []

# Loop through model configurations
for config in best_configs:
    model = config["model"]
    features = feature_sets[config["feature_set"]]
    task = config["task"].lower()
    threshold = config["threshold"]

    # Prepare training data for the current task
    df_train_task = df_train.copy()
    if task == "left_vs_rest":
        df_train_task = df_train_task[df_train_task['single_lat'].isin(['left', 'right', 'bilateral'])].copy()
        y_train = (df_train_task['single_lat'] == 'left').astype(int)
        target = "left"
    elif task == "right_vs_rest":
        df_train_task = df_train_task[df_train_task['single_lat'].isin(['left', 'right', 'bilateral'])].copy()
        y_train = (df_train_task['single_lat'] == 'right').astype(int)
        target = "right"
    else:
        continue

    X_train = df_train_task[features]

    # Fit the model (CRITICAL FIX)
    if len(np.unique(y_train)) > 1:
        model.fit(X_train, y_train)
    else:
        print(f"Skipping task {task}: Not enough classes in training data.")
        continue

    # Prepare data for prediction on both Engel and ILAE dataframes
    df_sub_engel = df_all_engel[df_all_engel["threshold"] == threshold].copy()
    df_sub_engel = df_sub_engel.drop_duplicates(subset="admission_id")
    df_sub_engel["single_lat"] = df_sub_engel["single_lat"].str.lower()
    df_sub_engel["surgery_side"] = df_sub_engel["single_lat"]
    df_sub_engel.loc[df_sub_engel["patient_id"] == 930, "surgery_side"] = "right"
    df_sub_engel = df_sub_engel[df_sub_engel["surgery_side"].isin(["left", "right"])]
    df_sub_engel = df_sub_engel[df_sub_engel["engel_binary"].isin([0, 1])]

    df_sub_ilae = df_all_ilae[df_all_ilae["threshold"] == threshold].copy()
    df_sub_ilae = df_sub_ilae.drop_duplicates(subset="admission_id")
    df_sub_ilae["single_lat"] = df_sub_ilae["single_lat"].str.lower()
    df_sub_ilae["surgery_side"] = df_sub_ilae["single_lat"]
    df_sub_ilae.loc[df_sub_ilae["patient_id"] == 930, "surgery_side"] = "right"
    df_sub_ilae = df_sub_ilae[df_sub_ilae["surgery_side"].isin(["left", "right"])]
    df_sub_ilae = df_sub_ilae[df_sub_ilae["ilae_binary"].isin([0, 1])]

    # Predict and compute concordance for Engel data
    for _, row in df_sub_engel.iterrows():
        # FIX: Ensure prediction data retains feature names
        x_patient = pd.DataFrame([row[features].values], columns=features)
        prob = model.predict_proba(x_patient)[0, 1]
        surgery_lat = row["surgery_side"]
        concordance_prob = prob if surgery_lat == target else 1 - prob
        predicted_lat = target if prob > 0.5 else "not_" + target
        concordance_data_engel.append({
            "patient_id": row.get("patient_id"),
            "admission_id": row["admission_id"],
            "task": task,
            "engel_binary": row["engel_binary"],
            "surgery_side": surgery_lat,
            "prob": prob,
            "concordance_prob": concordance_prob,
            "predicted_lat": predicted_lat
        })
    
    # Predict and compute concordance for ILAE data
    for _, row in df_sub_ilae.iterrows():
        # FIX: Ensure prediction data retains feature names
        x_patient = pd.DataFrame([row[features].values], columns=features)
        prob = model.predict_proba(x_patient)[0, 1]
        surgery_lat = row["surgery_side"]
        concordance_prob = prob if surgery_lat == target else 1 - prob
        predicted_lat = target if prob > 0.5 else "not_" + target
        concordance_data_ilae.append({
            "patient_id": row.get("patient_id"),
            "admission_id": row["admission_id"],
            "task": task,
            "ilae_binary": row["ilae_binary"],
            "surgery_side": surgery_lat,
            "prob": prob,
            "concordance_prob": concordance_prob,
            "predicted_lat": predicted_lat
        })

df_concordance_engel = pd.DataFrame(concordance_data_engel)
df_concordance_ilae = pd.DataFrame(concordance_data_ilae)

# ------------------------------------------------------------
# 4. Plot Engel (Fig 4B)
# ------------------------------------------------------------
# Combine left-vs-rest and right-vs-rest data
df_left_engel = df_concordance_engel[df_concordance_engel["task"] == "left_vs_rest"]
df_right_engel = df_concordance_engel[df_concordance_engel["task"] == "right_vs_rest"]
df_plot_engel = pd.concat([
    df_left_engel[df_left_engel["surgery_side"] == "left"],
    df_right_engel[df_right_engel["surgery_side"] == "right"]
], ignore_index=True)

# Add outcome labels and categorical order
df_plot_engel["outcome_label"] = df_plot_engel["engel_binary"].map({1: "Engel I", 0: "Engel II+"})
df_plot_engel["outcome_label"] = pd.Categorical(df_plot_engel["outcome_label"], categories=["Engel I", "Engel II+"], ordered=True)

# Mann-Whitney U test
group1_engel = df_plot_engel[df_plot_engel["engel_binary"] == 1]["concordance_prob"].to_numpy()
group2_engel = df_plot_engel[df_plot_engel["engel_binary"] == 0]["concordance_prob"].to_numpy()
u_stat_engel, p_u_engel = mannwhitneyu(group1_engel, group2_engel, alternative='two-sided')

print(f"Engel Concordance vs. Outcome: Mann-Whitney U statistic={u_stat_engel:.2f}, p={p_u_engel:.3g}")

# Create jittered scatter plot
np.random.seed(0)
x_map_engel = {"Engel I": 0, "Engel II+": 1}
df_plot_engel["x"] = df_plot_engel["outcome_label"].map(x_map_engel).astype(float)
df_plot_engel["x_jitter"] = df_plot_engel["x"] + np.random.uniform(-0.1, 0.1, size=len(df_plot_engel))
color_map = {'left': 'blue', 'right': 'red'}
df_plot_engel["dot_color"] = df_plot_engel["surgery_side"].map(color_map).fillna("gray")

plt.figure(figsize=(6, 5))
plt.scatter(
    df_plot_engel["x_jitter"],
    df_plot_engel["concordance_prob"],
    c=df_plot_engel["dot_color"],
    edgecolors="black",
    alpha=0.8,
    s=80
)
plt.xticks([0, 1], ["Engel I", "Engel II+"])
plt.xlim(-0.5, 1.5)
plt.ylim(0, 1.1)
plt.ylabel("Concordance Probability")
plt.title("Concordance vs. Surgical Outcome (Engel)")
plt.text(0.5, 1.02, f"p={p_u_engel:.3g}", ha='center', fontsize=11)
plt.grid(False)
plt.tight_layout()
plt.savefig(save_path_4B)
# plt.close()

# ------------------------------------------------------------
# 5. Plot ILAE (Fig 4D)
# ------------------------------------------------------------
# Combine left-vs-rest and right-vs-rest data
df_left_ilae = df_concordance_ilae[df_concordance_ilae["task"] == "left_vs_rest"]
df_right_ilae = df_concordance_ilae[df_concordance_ilae["task"] == "right_vs_rest"]
df_plot_ilae = pd.concat([
    df_left_ilae[df_left_ilae["surgery_side"] == "left"],
    df_right_ilae[df_right_ilae["surgery_side"] == "right"]
], ignore_index=True)

# Add outcome labels and categorical order
df_plot_ilae["outcome_label"] = df_plot_ilae["ilae_binary"].map({1: "ILAE good (1–2)", 0: "ILAE bad (3–5)"})
df_plot_ilae["outcome_label"] = pd.Categorical(
    df_plot_ilae["outcome_label"],
    categories=["ILAE good (1–2)", "ILAE bad (3–5)"],
    ordered=True
)

# Mann-Whitney U test
group_good_ilae = df_plot_ilae[df_plot_ilae["ilae_binary"] == 1]["concordance_prob"].to_numpy()
group_bad_ilae = df_plot_ilae[df_plot_ilae["ilae_binary"] == 0]["concordance_prob"].to_numpy()
u_stat_ilae, p_u_ilae = mannwhitneyu(group_good_ilae, group_bad_ilae, alternative='two-sided')

print(f"ILAE Concordance vs. Outcome: Mann-Whitney U statistic={u_stat_ilae:.2f}, p={p_u_ilae:.3g}")

# Create jittered scatter plot
np.random.seed(0)
x_map_ilae = {"ILAE good (1–2)": 0, "ILAE bad (3–5)": 1}
df_plot_ilae["x"] = df_plot_ilae["outcome_label"].map(x_map_ilae).astype(float)
df_plot_ilae["x_jitter"] = df_plot_ilae["x"] + np.random.uniform(-0.1, 0.1, size=len(df_plot_ilae))
color_map = {"left": "blue", "right": "red"}
df_plot_ilae["dot_color"] = df_plot_ilae["surgery_side"].map(color_map).fillna("gray")

plt.figure(figsize=(6, 5))
plt.scatter(
    df_plot_ilae["x_jitter"],
    df_plot_ilae["concordance_prob"],
    c=df_plot_ilae["dot_color"],
    edgecolors="black",
    alpha=0.8,
    s=80
)
plt.xticks([0, 1], ["ILAE 1-2", "ILAE 3+"])
plt.xlim(-0.5, 1.5)
plt.ylim(0, 1.1)
plt.ylabel("Concordance Probability")
plt.title("Concordance vs. Surgical Outcome (ILAE)")
plt.text(0.5, 1.02, f"p={p_u_ilae:.3g}", ha="center", fontsize=11)
plt.grid(False)
plt.tight_layout()
plt.savefig(save_path_4D)
# plt.close()
