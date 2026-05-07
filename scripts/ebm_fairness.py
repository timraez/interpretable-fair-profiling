import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

import os

def plot_confusion_matrices_by_group(conf_matrices, title):
    # Create results folder if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    label_map = [['TP', 'FP'],
                 ['FN', 'TN']]
    
    for group, matrix in conf_matrices.items():
        # build annotations "TP\n0.42" etc.
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
        plt.title(f'CM {title} — Group: {group}')
        plt.xlabel('Actual label')
        plt.ylabel('Predicted label')
        plt.tight_layout()
        
        # Save the plot
        filename = f"cm_{title}_{group}.png"
        filepath = os.path.join('results', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        plt.show()
        plt.close()  # Close the figure to free memory


from fairlearn.postprocessing import ThresholdOptimizer

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


def add_group_intersection_column(df, new_column_name='gruppen'):

    # Convert alter to string (handle categorical if needed)
    alter_str = df['alter_gruppen'].astype(object).astype(str)
    
    # Recode cat_geschlecht: 0 -> 'f', 1 -> 'm'
    geschlecht_map = {0: 'f', 1: 'm'}
    geschlecht_recoded = df['cat_geschlecht'].map(geschlecht_map).astype(str)
    
    # Recode aufenthalt_bins: 0 -> 'pr', 1 -> 'npr'
    aufenthalt_map = {0: 'pr', 1: 'npr'}
    aufenthalt_recoded = df['aufenthalt_bins'].map(aufenthalt_map).astype(str)
    
    # Create the combined column
    df[new_column_name] = (
        'age=' + alter_str + 
        '_gen=' + geschlecht_recoded + 
        '_res=' + aufenthalt_recoded
    )
    return df


def add_groups_fold(fold):
    new_fold = fold.copy()

    holdout_X = new_fold['holdout_X']

    holdout_X_ag = add_group_intersection_column(holdout_X)

    fold['holdout_X'] = holdout_X_ag

    group_fractions = holdout_X_ag['gruppen'].value_counts(normalize=False)

    return fold, group_fractions


def get_incidence_rate_intersections(holdout_X, holdout_Y):
    holdout_X_Y = pd.merge(holdout_X, holdout_Y, left_index=True, right_index=True)

    # Group by 'gruppen' and calculate the mean of the 'target' variable
    incidence_rates = holdout_X_Y.groupby('gruppen', observed=True)['f01_zug_lz_al_12_w2z'].sum()

    return incidence_rates

def get_incidence_rate_intersections_fold(fold):
    new_fold = fold.copy()

    holdout_X = new_fold['holdout_X']
    holdout_Y = new_fold['holdout_Y']

    incidence_rates = get_incidence_rate_intersections(holdout_X, holdout_Y)

    return incidence_rates


def get_postprocess_fit_predict_g(fold):
    new_fold = fold.copy()
    
    X_train = fold['CV_X'].copy(deep=True)
    y_train = fold['CV_Y'].copy(deep=True)
    holdout_X = fold['holdout_X'].copy()
    
    # Define bin edges
    bins = [15, 30, 45, 66]
    # Define labels for the bins
    labels = ['15-29', '30-44', '45-65']
    
    # Apply pd.cut to create a new column with the specified bins
    age_train = pd.cut(X_train['alter'], bins=bins, labels=labels, right=False).astype(object).astype(str)
    age_test = pd.cut(holdout_X['alter'], bins=bins, labels=labels, right=False).astype(object).astype(str)
    
    # Recode cat_geschlecht: 0 -> 'f', 1 -> 'm'
    geschlecht_map = {0: 'f', 1: 'm'}
    gen_train = X_train['cat_geschlecht'].map(geschlecht_map).astype(str)
    gen_test = holdout_X['cat_geschlecht'].map(geschlecht_map).astype(str)
    
    # Recode aufenthalt_bins: 0 -> 'pr', 1 -> 'npr'
    aufenthalt_map = {0: 'pr', 1: 'npr'}
    res_train = X_train['aufenthalt_bins'].map(aufenthalt_map).astype(str)
    res_test = holdout_X['aufenthalt_bins'].map(aufenthalt_map).astype(str)
    
    # Create the combined column
    A_train = (
        'age=' + age_train + 
        '_gen=' + gen_train + 
        '_res=' + res_train
    )
    
    A_test = (
        'age=' + age_test + 
        '_gen=' + gen_test + 
        '_res=' + res_test
    )
    
    # Remove any NaNs - check all components that could have NaNs
    mask = (
        age_train.notna() & 
        (age_train != 'nan') &  # pd.cut can create 'nan' strings
        gen_train.notna() & 
        res_train.notna() &
        ~A_train.str.contains('nan', na=False)  # Extra safety check
    )
    
    X_train_clean = X_train.loc[mask]
    y_train_clean = y_train.loc[mask]
    A_train_clean = A_train.loc[mask]
    
    ebm = fold['model']
    postprocess = ThresholdOptimizer(
        estimator=ebm,
        constraints="false_positive_rate_parity",
        objective="balanced_accuracy_score",
        predict_method="predict_proba",
        prefit=True
    )
    
    postprocess.fit(X_train_clean, y_train_clean, sensitive_features=A_train_clean)
    new_fold['model_fair'] = postprocess
    y_preds_fair = postprocess.predict(holdout_X, sensitive_features=A_test)
    new_fold['y_preds_fair'] = y_preds_fair
    
    return new_fold

