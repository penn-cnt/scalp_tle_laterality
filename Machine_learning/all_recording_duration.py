import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# === Constants ===
durations_min = list(range(30, 1801, 30))

# === File Paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
train_folder = os.path.join(script_dir, "/Dataset/patient_sai_values_all_training.csv")
test_folder = os.path.join(script_dir, "/Dataset/patient_sai_values_all_test.csv")
output_dir = os.path.join(script_dir, "/Visualization/")
train_folder = pd.read_csv(train_folder)
test_folder = pd.read_csv(test_folder)

def bootstrap_auc(y_true, y_score, n_iterations=1000, alpha=0.05):
    """Calculates bootstrap 95% CI for AUC and returns the median AUC."""
    aucs = []
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    for _ in range(n_iterations):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        try:
            if len(np.unique(y_true[indices])) < 2:
                continue
            auc = roc_auc_score(y_true[indices], y_score[indices])
            aucs.append(auc)
        except ValueError:
            continue
    
    if aucs:
        median_auc = np.median(aucs)
        lower = np.percentile(aucs, 100 * alpha / 2)
        upper = np.percentile(aucs, 100 * (1 - alpha / 2))
        return median_auc, lower, upper
    else:
        return np.nan, np.nan, np.nan

def label_logic(task, lat):
    if task == 'left':
        return 1 if lat == 'left' else 0
    elif task == 'right':
        return 1 if lat == 'right' else 0
    return None

def train_and_test_model_by_duration(task, train_df, test_df):
    aucs, n_subjects, ci_lowers, ci_uppers = [], [], [], []
    
    time_points = list(range(30, 1801, 30))
    
    for d_min in time_points:
        # Select columns for the current duration
        cols_to_use = [f'SAI_{d_min}min']
        print(f"\n--- Processing {d_min} min duration for task '{task.upper()}' ---")
        
        # --- Training ---
        # Dropping rows with any NaN values in the columns of interest
        train_subset = train_df.dropna(subset=cols_to_use)
        
        if train_subset.empty:
            print(f"Skipping {d_min}min training for {task} due to insufficient data.")
            aucs.append(np.nan)
            n_subjects.append(0)
            ci_lowers.append(np.nan)
            ci_uppers.append(np.nan)
            continue
        
        X_train = train_subset[cols_to_use].values
        y_train = train_subset['Lateralization'].apply(lambda x: label_logic(task, x)).values
        
        if len(np.unique(y_train)) < 2:
            print(f"Skipping {d_min}min training for {task} due to only one class present.")
            aucs.append(np.nan)
            n_subjects.append(len(train_subset))
            ci_lowers.append(np.nan)
            ci_uppers.append(np.nan)
            continue
            
        # Using a fixed LogisticRegression model
        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        print(f"Model trained with {len(y_train)} subjects for {d_min} min duration.")

        # --- Testing ---
        # Dropping rows with any NaN values in the columns of interest for the test set
        test_subset = test_df.dropna(subset=cols_to_use).copy()
        
        if test_subset.empty:
            print(f"Skipping {d_min}min testing for {task} due to insufficient data.")
            aucs.append(np.nan)
            n_subjects.append(0)
            ci_lowers.append(np.nan)
            ci_uppers.append(np.nan)
            continue

        X_test = test_subset[cols_to_use].values
        y_test = test_subset['Lateralization'].apply(lambda x: label_logic(task, x)).values
        
        valid_indices = y_test != None
        X_test = X_test[valid_indices]
        y_test = y_test[valid_indices]

        if X_test.size == 0 or len(np.unique(y_test)) < 2:
            print(f"Skipping {d_min}min testing for {task} due to insufficient classes in test data.")
            aucs.append(np.nan)
            n_subjects.append(len(test_subset))
            ci_lowers.append(np.nan)
            ci_uppers.append(np.nan)
        else:
            try:
                prob = model.predict_proba(X_test)[:, 1]
                median_auc, lower, upper = bootstrap_auc(y_test, prob)
                aucs.append(median_auc)
                n_subjects.append(len(test_subset))
                ci_lowers.append(lower)
                ci_uppers.append(upper)
                print(f"  --> AUC for {d_min} min: {median_auc:.3f} | CI: ({lower:.3f}, {upper:.3f}) with {len(y_test)} subjects.")
            except ValueError:
                aucs.append(np.nan)
                n_subjects.append(len(test_subset))
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
    
    return aucs, n_subjects, ci_lowers, ci_uppers

# === Main Execution Flow ===
if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    
    # Load feature and label data from the specified CSVs
    train_features_df = pd.read_csv(train_folder)
    test_features_df = pd.read_csv(test_folder)
    
    all_auc_results = {}
    for task in ['left', 'right']:
        print(f"\n--- Running full analysis for {task.upper()} vs rest ---")
        # Pass the dataframes directly, as they contain all necessary columns
        aucs, counts, ci_lowers, ci_uppers = train_and_test_model_by_duration(task, train_features_df, test_features_df)
        all_auc_results[task] = (durations_min, aucs, counts, ci_lowers, ci_uppers)

    print("\n--- Final Results ---")
    for task, results in all_auc_results.items():
        dur_hr = [d / 60 for d in results[0]]
        aucs = results[1]
        
        print(f"\nTask: {task.capitalize()} vs rest")
        for i, auc in enumerate(aucs):
            if not np.isnan(auc):
                print(f"  Duration: {dur_hr[i]:.1f} hr | AUC: {auc:.3f} | 95% CI: ({results[3][i]:.3f} - {results[4][i]:.3f})")

    plt.figure(figsize=(8, 5))
    for task in ['left', 'right']:
        dur_hr = [d / 60 for d in all_auc_results[task][0]]
        aucs = all_auc_results[task][1]
        ci_lowers = all_auc_results[task][3]
        ci_uppers = all_auc_results[task][4]
        
        plt.plot(dur_hr, aucs, marker='o', label=f"{task.capitalize()} vs rest")
        plt.fill_between(dur_hr, ci_lowers, ci_uppers, alpha=0.2)

    plt.xlabel("Duration (hr)")
    plt.ylabel("Test AUC")
    plt.title("AUC vs Duration on Test Set with 95% CI (Train by Duration)")
    plt.xticks(np.arange(0, 31, 5))
    plt.grid(True)
    plt.ylim(0.4, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3C.jpg"))
    # plt.close()