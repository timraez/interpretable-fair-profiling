import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

def add_age_group_column(df):

    # Define bin edges
    bins = [15, 30, 45, 66]

    # Define labels for the bins
    labels = ['15-29', '30-44', '45-65']

    # Apply pd.cut to create a new column with the specified bins
    df['alter_gruppen'] = pd.cut(df['alter'], bins=bins, labels=labels, right=False)
    return df

def add_age_group_fold(fold):
    new_fold = fold.copy()

    holdout_X = new_fold['holdout_X']

    holdout_X_ag = add_age_group_column(holdout_X)

    fold['holdout_X'] = holdout_X_ag

    group_fractions = holdout_X_ag['alter_gruppen'].value_counts(normalize=True)

    return fold, group_fractions

def get_incidence_rate(holdout_X, holdout_Y):
    holdout_X_Y = pd.merge(holdout_X, holdout_Y, left_index=True, right_index=True)

    # Group by 'age_group' and calculate the mean of the 'target' variable
    incidence_rates = holdout_X_Y.groupby('alter_gruppen', observed=True)['f01_zug_lz_al_12_w2z'].mean()

    return incidence_rates

def get_incidence_rate_fold(fold):
    new_fold = fold.copy()

    holdout_X = new_fold['holdout_X']
    holdout_Y = new_fold['holdout_Y']

    incidence_rates = get_incidence_rate(holdout_X, holdout_Y)

    return incidence_rates


def compute_pred_probs(fold):
    new_fold = fold.copy()
    holdout_X = fold['holdout_X']
    holdout_X = holdout_X.copy()
  
    ebm = fold['model']
        
    y_probs = ebm.predict_proba(holdout_X)[:, 1]

    new_fold['pred_probs'] = y_probs

    return new_fold

def plot_cumulative_gain_curve(fold, pos_label=1, n_bins=100):
    """
    Plots a cumulative gain curve.
    
    Parameters:
    - pos_label: label considered positive
    - n_bins: resolution of the curve (number of cutoff points)
    """
    y_true = fold['holdout_Y']
    y_probs = fold['pred_probs']

    # Step 1: Sort by predicted probabilities in descending order
    data = pd.DataFrame({'y_true': y_true, 'y_prob': y_probs})
    data = data.sort_values('y_prob', ascending=False).reset_index(drop=True)

    # Step 2: Compute cumulative true positives
    data['cum_positives'] = (data['y_true'] == pos_label).cumsum()
    total_positives = data['y_true'].sum()

    # Step 3: Sample cutoff points
    sample_points = np.linspace(0, len(data) - 1, n_bins + 1, dtype=int)
    x_vals = sample_points / len(data)  # share of population
    y_vals = data['cum_positives'].iloc[sample_points].fillna(method='ffill') / total_positives  # share of positives

    # Step 4: Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='Model', color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    plt.xlabel('Share of Population Targeted')
    plt.ylabel('Share of Positive Cases Captured')
    plt.title('Cumulative Gain Curve')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(which='major', color='gray', linestyle='-', linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def population_share_for_recall(fold, target_recall=0.8, pos_label=1):
    # Step 1: Sort by predicted probability descending

    y_true = fold['holdout_Y']
    y_probs = fold['pred_probs']

    data = pd.DataFrame({'y_true': y_true, 'y_prob': y_probs})
    data = data.sort_values('y_prob', ascending=False).reset_index(drop=True)
    
    # Step 2: Convert y_true to binary
    data['y_true'] = (data['y_true'] == pos_label).astype(int)
    
    # Step 3: Compute cumulative true positives
    data['cum_positives'] = data['y_true'].cumsum()
    total_positives = data['y_true'].sum()
    data['recall'] = data['cum_positives'] / total_positives
    
    # Step 4: Compute population percent
    data['population_frac'] = (np.arange(1, len(data)+1)) / len(data)
    
    # Step 5: Find smallest population fraction where recall >= target_recall
    idx = data[data['recall'] >= target_recall].index[0]
    pop_fraction_needed = data.loc[idx, 'population_frac']
    
    return pop_fraction_needed


def add_binary_risk_column(fold, quant):
    """
    Adds a binary risk group column to the DataFrame based on a probability threshold.

    Parameters:
    - df: pandas DataFrame with features.
    - model: trained model with predict_proba method.
    - threshold: float, cutoff for assigning risk group 1.

    Returns:
    - DataFrame with added 'risikogruppe' column (0 or 1).
    """
    new_fold = fold.copy()
    holdout_X = fold['holdout_X']
    holdout_X = holdout_X.copy()
  
    model = fold['model']
    
    # Get predicted probabilities for the positive class
    risk_scores = model.predict_proba(holdout_X)[:, 1]

    risk_scores = pd.Series(risk_scores, index=holdout_X.index)

    threshold = risk_scores.quantile(quant)

    # Assign 1 if score >= threshold, else 0
    risk_bins = (risk_scores >= threshold).astype(int)

    # Convert to Series and name it
    risk_bins = pd.Series(risk_bins, index=holdout_X.index, name='risk_bins')

    new_fold['y_preds'] = risk_bins

    return new_fold, threshold


def get_normalized_confusion_matrices_by_group(fold, prediction, sensitive_attribute):
    """
    Returns normalized confusion matrices for each group in a sensitive attribute,
    reordered to match the custom layout:
    [[TP, FP],
     [FN, TN]]
    """
    y_true = fold['holdout_Y']
    y_pred = fold[prediction]
    sensitive_attribute = fold['holdout_X'][sensitive_attribute]

    results = {}
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group': sensitive_attribute
    })

    for group in df['group'].unique():
        group_df = df[df['group'] == group]
        # Compute normalized confusion matrix
        cm = confusion_matrix(group_df['y_true'], group_df['y_pred'], labels=[0, 1], normalize='all')
        # Reorder to custom layout: [[TP, FP], [FN, TN]]
        TP = cm[1, 1]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TN = cm[0, 0]
        cm_custom = np.array([[TP, FP], [FN, TN]])
        results[group] = cm_custom

    return results

def plot_confusion_matrices_by_group(conf_matrices, title):
    """
    Plots labeled normalized confusion matrices for each group, with:
      cm_custom = [[TP, FP],
                   [FN, TN]]
    and axes:
      rows → Predicted (Positive, Negative)
      cols → Actual    (Positive, Negative)
    """
    label_map = [['TP', 'FP'],
                 ['FN', 'TN']]

    for group, matrix in conf_matrices.items():
        # build annotations “TP\n0.42” etc.
        annotations = np.empty_like(matrix, dtype=object)
        for i in (0,1):
            for j in (0,1):
                annotations[i, j] = f"{label_map[i][j]}\n{matrix[i,j]:.2f}"

        plt.figure(figsize=(5,4))
        sns.heatmap(
            matrix,
            annot=annotations,
            fmt='',
            cmap='Blues',
            vmin=0, vmax=1,
            xticklabels=['Actual Positive', 'Actual Negative'],
            yticklabels=['Predicted Positive', 'Predicted Negative']
        )
        plt.title(f'Normalized Confusion Matrix {title} — Group: {group}')
        plt.xlabel('Actual label')
        plt.ylabel('Predicted label')
        plt.tight_layout()
        plt.show()


from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer

def get_postprocess_fit_predict(fold):
    new_fold = fold.copy()
    
    X_train = fold['CV_X'].copy(deep=True)
    y_train = fold['CV_Y'].copy(deep=True)

    holdout_X = fold['holdout_X']
    holdout_X = holdout_X.copy()

    # Define bin edges
    bins = [15, 30, 45, 66]

    # Define labels for the bins
    labels = ['15-29', '30-44', '45-65']

    # Apply pd.cut to create a new column with the specified bins
    A_train = pd.cut(X_train['alter'], bins=bins, labels=labels, right=False)

    A_test = pd.cut(holdout_X['alter'], bins=bins, labels=labels, right=False)

    # Remove any NaNs in sensitive feature before fitting
    mask = A_train.notna()
    X_train_clean = X_train.loc[mask]
    y_train_clean = y_train.loc[mask]
    A_train_clean = A_train.loc[mask]

    ebm = fold['model']

    postprocess = ThresholdOptimizer(
    estimator=ebm,
    constraints="false_positive_rate_parity",  # enforce equal FPR
    objective="balanced_accuracy_score",       # optional performance objective
    predict_method="predict_proba",
    prefit=True                                # our estimator is already fitted
    )

    postprocess.fit(X_train_clean, y_train_clean, sensitive_features=A_train_clean)

    new_fold['model_fair'] = postprocess

    y_preds_fair = postprocess.predict(holdout_X, sensitive_features=A_test)

    new_fold['y_preds_fair'] = y_preds_fair

    return new_fold


from sklearn.metrics import confusion_matrix, accuracy_score

def plot_tpr_positive_accuracy(y_true, y_pred, title):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    TN, FP, FN, TP = cm.ravel()

    # Compute metrics
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    proportion_positive = sum(y_pred == 1) / len(y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Prepare data for plotting
    metrics = {
        "True Positive Rate": tpr,
        "Proportion Positive": proportion_positive,
        "Accuracy": accuracy
    }

    # Plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics.keys(), metrics.values(), color=['tab:blue', 'tab:orange', 'tab:green'])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"Model Performance Metrics: {title}")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}", 
                 ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
