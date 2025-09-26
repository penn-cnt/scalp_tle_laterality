import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats
import os

warnings.filterwarnings("ignore")

# --- 1. BOOTSTRAPPING HELPER FUNCTION ---
def bootstrap_ci_and_scores(y_true, y_pred, metric_func, n_bootstraps=1000, confidence_level=0.95):
    """Calculates the bootstrap 95% CI and returns the scores."""
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true.values[indices])) < 2:
            continue
        
        score = metric_func(y_true.values[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        
    alpha = (1.0 - confidence_level) / 2.0
    lower_bound = np.percentile(bootstrapped_scores, alpha * 100)
    upper_bound = np.percentile(bootstrapped_scores, (1.0 - alpha) * 100)
    
    return bootstrapped_scores, lower_bound, upper_bound

# --- 2. MAKE BINARY LABEL HELPER FUNCTION ---
def make_binary_label(df, task):
    if task == 'left_vs_rest':
        return (df['single_lat'].str.lower().str.strip() == 'left').astype(int)
    elif task == 'right_vs_rest':
        return (df['single_lat'].str.lower().str.strip() == 'right').astype(int)

# --- 3. DeLong's Test Helper Functions (provided by user) ---
def compute_midrank(x):
    """Computes midranks."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    """The fast version of DeLong's method for computing the covariance of unadjusted AUC."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    """Computes p-value for hypothesis that two ROC AUCs are different."""
    l = np.array([[1, -1]])
    # Z = |AUC1 - AUC2| / SE_diff
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    p_value = 2.0 * scipy.stats.norm.sf(z, loc=0, scale=1)[0][0] 
    
    return p_value

def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """Computes log(p-value) for hypothesis that two ROC AUCs are different."""
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return aucs, delongcov, calc_pvalue(aucs, delongcov)

# --- 4. DATA LOADING AND FILTERING ---
script_dir = os.path.dirname(os.path.abspath(__file__))
df_train = os.path.join(script_dir, "sleep_wake_training_261_laterality.csv")
df_test = os.path.join(script_dir, "sleep_wake_test_104_laterality.csv")
df_train = pd.read_csv(df_train)
df_test = pd.read_csv(df_test)
SPIKE_RATE_THRESHOLD = 0.017

# Filter dataframes by threshold 0.43 and filter data who had more than a spike per hour in wake and sleep
df_train_filtered = df_train[df_train['threshold'] == 0.43].copy()
df_test_filtered = df_test[df_test['threshold'] == 0.43].copy()

df_train_filtered = df_train_filtered[
    (df_train_filtered['spike_rate_wake_min'] >= SPIKE_RATE_THRESHOLD) &
    (df_train_filtered['spike_rate_sleep_min'] >= SPIKE_RATE_THRESHOLD)
].copy()

df_test_filtered = df_test_filtered[
    (df_test_filtered['spike_rate_wake_min'] >= SPIKE_RATE_THRESHOLD) &
    (df_test_filtered['spike_rate_sleep_min'] >= SPIKE_RATE_THRESHOLD)
].copy()

# --- 5. DATA PREPARATION FOR DeLONG'S TEST ---
# Create a single, consistent test set for all comparisons
# This is the crucial fix. It ensures that both SAI_wake and SAI_sleep values are present.
required_features = ['SAI_wake', 'SAI_sleep', 'single_lat', 'patient_id']
df_test_consistent = df_test_filtered.dropna(subset=required_features).copy()
df_test_consistent = df_test_consistent.sort_values('patient_id').reset_index(drop=True)

print(f"Number of patients in the consistent test set: {len(df_test_consistent)}")


# --- 6. EVALUATE FOUR SCENARIOS ---
results = {}
predictions = {}
scenarios = [
    {'task': 'left_vs_rest', 'state': 'wake', 'feature': 'SAI_wake', 'label': 'Wake L vs R/BL'},
    {'task': 'left_vs_rest', 'state': 'sleep', 'feature': 'SAI_sleep', 'label': 'Sleep L vs R/BL'},
    {'task': 'right_vs_rest', 'state': 'wake', 'feature': 'SAI_wake', 'label': 'Wake R vs L/BL'},
    {'task': 'right_vs_rest', 'state': 'sleep', 'feature': 'SAI_sleep', 'label': 'Sleep R vs L/BL'}
]

print("\n--- Test AUC (Median) and 95% CI ---")
for scenario in scenarios:
    task = scenario['task']
    feature_col = scenario['feature']
    label = scenario['label']
    
    # Use the consistent test data for evaluation, but use all available training data
    df_train_sub = df_train_filtered.dropna(subset=[feature_col, 'single_lat']).copy()
    
    X_train = df_train_sub[[feature_col]]
    y_train = make_binary_label(df_train_sub, task)
    
    # Use the consistent test set here
    X_test = df_test_consistent[[feature_col]]
    y_test = make_binary_label(df_test_consistent, task)

    if y_test.nunique() < 2:
        print(f"Skipping {label}: Test set has only one class.")
        continue

    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    bootstrapped_auc_scores, ci_low, ci_high = bootstrap_ci_and_scores(y_test, y_test_prob, roc_auc_score)
    median_auc = np.median(bootstrapped_auc_scores)

    print(f" {label}:")
    print(f"  Test AUC (Median): {median_auc:.4f}")
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print("-" * 25)
    
    results[label] = {
        'auc_scores': bootstrapped_auc_scores,
        'ci_low': ci_low,
        'ci_high': ci_high
    }

    # Store the predictions and true labels from the consistent test set
    predictions[label] = {
        'y_true': y_test.values,
        'y_pred_prob': y_test_prob
    }


# --- 7. PERFORM DeLONG'S TEST FOR AUC COMPARISONS ---
print("\n--- DeLong's Test p-values for AUC comparisons ---")

def run_delongs_test(predictions, label1, label2):
    """Runs DeLong's test for a given pair of labels."""
    if label1 not in predictions or label2 not in predictions:
        print(f"Skipping comparison: Missing data for {label1} or {label2}")
        return

    y_true_1 = predictions[label1]['y_true']
    y_pred_1 = predictions[label1]['y_pred_prob']
    y_true_2 = predictions[label2]['y_true']
    y_pred_2 = predictions[label2]['y_pred_prob']
    
    # This check will now pass because both y_true arrays come from the same consistent dataset
    if not np.array_equal(y_true_1, y_true_2):
        print(f"Warning: The ground truths for {label1} and {label2} are not identical. Skipping comparison.")
        return

    if len(np.unique(y_true_1)) < 2:
        print(f"Warning: The ground truth for {label1} and {label2} has only one class. Skipping comparison.")
        return

    try:
        aucs, _, p_value_log10 = delong_roc_test(y_true_1, y_pred_1, y_pred_2)
        p_value = np.exp(np.log(10) * p_value_log10)[0][0]

        print(f"  {label1} vs {label2}:")
        print(f"    AUC ({label1}): {aucs[0]:.4f}")
        print(f"    AUC ({label2}): {aucs[1]:.4f}")
        print(f"    p-value = {p_value:.6f}")
        if p_value < 0.05:
            print("    Result: The difference in AUC is statistically significant (p < 0.05).")
        else:
            print("    Result: The difference in AUC is not statistically significant (p >= 0.05).")
        print("-" * 25)

    except Exception as e:
        print(f"    Could not perform DeLong's test for {label1} vs {label2}: {e}")

# Left vs Rest Task: Compare Wake and Sleep
run_delongs_test(predictions, 'Wake L vs R/BL', 'Sleep L vs R/BL')

# Right vs Rest Task: Compare Wake and Sleep
run_delongs_test(predictions, 'Wake R vs L/BL', 'Sleep R vs L/BL')


# --- 8. PLOT AND SAVE THE RESULTS (Fig3D)---
if results:
    labels = list(results.keys())
    data_to_plot = [results[label]['auc_scores'] for label in labels]

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor('white')

    colors = ['royalblue', 'royalblue', 'darkorange', 'darkorange']

    violin_parts = ax.violinplot(data_to_plot, showmeans=False, showmedians=False)
    for i, part in enumerate(violin_parts['bodies']):
        part.set_facecolor(colors[i])
        part.set_edgecolor('black')
        part.set_alpha(0.7)

    for i, label in enumerate(labels):
        x_positions = np.random.normal(i + 1, 0.04, size=len(results[label]['auc_scores']))
        ax.scatter(
            x_positions,
            results[label]['auc_scores'],
            alpha=0.2,
            s=10,
            color=colors[i],
            edgecolors='none'
        )
        ax.scatter(i + 1, np.median(results[label]['auc_scores']), color='black', marker='o', s=50, zorder=3)
        ax.vlines(i + 1, results[label]['ci_low'], results[label]['ci_high'], color='black', linestyle='--', linewidth=2, zorder=2)

    x_positions = range(1, len(labels) + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=12)

    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('Test AUC Score', fontsize=16)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='white')

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', markersize=10, label='Left vs rest'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=10, label='Right vs rest')
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=12)
    
    plt.grid(True)
    plt.tight_layout()

    # Define the output directory relative to the script's location
    output_dir = os.path.join(script_dir, "Visualization") 
    output_filename = "Fig3D.jpg"
    
    # Save the figure to the specified path
    plt.savefig(os.path.join(output_dir, output_filename))
    print(f"\nFigure saved to: {os.path.join(output_dir, output_filename)}")
    
    # Close the plot to free up memory
    # plt.close()

