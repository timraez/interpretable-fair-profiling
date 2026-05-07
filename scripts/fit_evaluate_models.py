from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import copy
import pandas as pd
import itertools
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.utils import resample

def fit_model(folds, model_class, model_params=None):
    trained_folds = []
    for fold in folds:
        fold_copy = fold.copy()

        CV_X = fold_copy['CV_X'].copy()
        CV_Y = fold_copy['CV_Y'].copy()

        model = model_class(**model_params)
        model.fit(CV_X, CV_Y)

        fold_copy['model'] = model
        trained_folds.append(fold_copy)
    return trained_folds

def get_results(folds):
    scores_folds = []
    for fold in folds:
        holdout_X = fold['holdout_X']
        holdout_Y = fold['holdout_Y']

        model = fold['model']
        probabilities = model.predict_proba(holdout_X)[:, 1]

        auc_score = roc_auc_score(holdout_Y, probabilities)
        scores_folds.append(auc_score)
    return scores_folds

# feature information for lr (features to be normalized)
dataset_info = {'numerical_attributes': ['vers_verdienst', 'beschaeftigungsgrad_vorher', 'vermittlungsgrad_asal', 
'beitragsmonate_vor_rf', 'anz_b_ausgeuebt', 'anz_b_n_ausgeuebt', 'anz_b_gesucht', 
'anz_b_erf_3j', 'anz_b_erf_1bis3j', 'anz_b_erf_1j', 'anz_b_erf_0', 'anz_b_mit_erf_such',
'b_ges_ausg_sek', 'b_ges_ausg_ter', 'anz_b_gesucht_zuletzt', 'berufskl_1', 'berufskl_2',
'berufskl_3', 'berufskl_4', 'berufskl_5', 'berufskl_6', 'berufskl_7', 'berufskl_8',
'berufskl_9', 'anz_b_ges_gelernt', 'anz_b_ges_exp_quali', 'anz_b_ges_exp_lehre', 
'anz_b_ges_inl_abs', 'anz_b_ges_ausl_abs', 'alter']}

# fit logistic regression to folds
def fit_lr(folds, dataset_info, parameter):
    copied_folds = [copy.deepcopy(fold) for fold in folds]  # deep copy to avoid mutation
    for fold in copied_folds:
        CV_X = fold['CV_X']
        CV_Y = fold['CV_Y']
        
        scaler = StandardScaler()
        numerical = dataset_info['numerical_attributes']
        CV_X = CV_X.copy()
        CV_X[numerical] = scaler.fit_transform(CV_X[numerical])
        
        lr = LogisticRegression(penalty='l1', solver='liblinear', C=parameter).fit(CV_X, CV_Y)
        fold['model'] = lr
        fold['scaler'] = scaler
        fold['parameter'] = parameter
    return copied_folds

# get predictions and probabilites for the five test folds
def get_results_lr(folds, dataset_info):
    weights_folds = []
    scores_folds = []
    for fold in folds:
        holdout_X = fold['holdout_X']
        holdout_X = holdout_X.copy()

        holdout_Y = fold['holdout_Y']

        scaler = fold['scaler']  
        lr = fold['model']
        numerical = dataset_info['numerical_attributes']
        
        holdout_X[numerical] = scaler.transform(holdout_X[numerical])
        
        probabilities = lr.predict_proba(holdout_X)[:, 1]
        
        auc_score = roc_auc_score(holdout_Y, probabilities)
        scores_folds.append(auc_score)

        weights = lr.coef_[0]
        weights_folds.append(weights)
    return weights_folds, scores_folds

def save_folds_with_models(folds, filepath):
    joblib.dump(folds, filepath)

def load_folds_with_models(filepath):
    return joblib.load(filepath)

def combine_model_scores(scores_lr, scores_rf, scores_xgb, scores_gb, scores_ebm):

    df = pd.DataFrame({
        'LR': scores_lr,
        'RF': scores_rf,
        'XGB': scores_xgb,
        'GB': scores_gb,
        'EBM': scores_ebm
    })
    return df

def plot_model_scores(df, years, title='Model Performance Across Folds'):

    assert len(df) == len(years), "Length of 'years' must match number of folds in df."

    df.index.name = 'fold'

    markers = itertools.cycle(['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h', '<', '>'])

    fig, ax = plt.subplots(figsize=(8, 5))

    for column, marker in zip(df.columns, markers):
        ax.plot(range(len(df)), df[column], marker=marker, linestyle='dashed', label=column)

    # Set year labels on x-axis
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years])

    ax.set_xlabel('Validate Year')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(title='Model')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_model_scores_total(df, test_years, title='Model Performance Across Folds'):
    """
    Plot model scores with custom x-axis labels (years), and highlight final test year.
    
    Parameters:
    - df: pd.DataFrame, model scores per fold (rows = folds, columns = models)
    - test_years: list of int or str, e.g. [2015, 2016, 2017, 2018, 2019]
    - title: str, plot title
    """
    # Ensure index is named 'fold'
    df.index.name = 'fold'

    # Distinct markers for each model
    markers = itertools.cycle(['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h', '<', '>'])

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    for column, marker in zip(df.columns, markers):
        ax.plot(range(len(df)), df[column], marker=marker, linestyle='dashed', label=column)

    # Set x-tick labels to test years
    x_positions = list(range(len(test_years)))
    x_labels = [str(y) for y in test_years]

    # Create custom tick labels, bold for final year
    xtick_labels = []
    for i, label in enumerate(x_labels):
        if i == len(x_labels) - 1:
            label = f"$\\bf{{{label}}}$"  # Render final year in bold
        xtick_labels.append(label)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels)

    # Add vertical dashed line before the final test year
    if len(test_years) > 1:
        ax.axvline(x=len(test_years) - 1.5, linestyle='--', color='gray', linewidth=1)

    # Labels and title
    ax.set_xlabel('Validate and Test Year')
    ax.set_ylabel('Score')
    ax.set_title(title)

    ax.legend(title='Model')
    ax.grid(True)
    plt.tight_layout()
    plt.show()



def compare_models_paired_wilcoxon(folds_model1, folds_model2, n_bootstrap=500, random_state=42):

    np.random.seed(random_state)
    comparison_results = []
    
    for fold_idx, (fold1, fold2) in enumerate(zip(folds_model1, folds_model2)):
        # Verify same holdout set size and outcomes
        assert len(fold1['holdout_Y']) == len(fold2['holdout_Y']), \
            f"Fold {fold_idx}: Holdout sets must be the same size"
        
        # Each model has its own X (different encodings), but same Y
        holdout_X1 = fold1['holdout_X']
        holdout_X2 = fold2['holdout_X']
        holdout_Y = fold1['holdout_Y']  # Should be identical to fold2['holdout_Y']
        model1 = fold1['model']
        model2 = fold2['model']
        
        # Store paired bootstrap AUC scores
        bootstrap_aucs_model1 = []
        bootstrap_aucs_model2 = []
        
        # Perform stratified bootstrap with SAME indices for both models
        for i in range(n_bootstrap):
            # Generate indices once - this is what makes the samples paired
            indices = resample(
                np.arange(len(holdout_Y)),
                stratify=holdout_Y,
                replace=True,
                random_state=random_state + i
            )
            
            # Apply same indices to each model's X (but they have different features)
            X_bootstrap1 = holdout_X1.iloc[indices] if hasattr(holdout_X1, 'iloc') else holdout_X1[indices]
            X_bootstrap2 = holdout_X2.iloc[indices] if hasattr(holdout_X2, 'iloc') else holdout_X2[indices]
            y_bootstrap = holdout_Y.iloc[indices] if hasattr(holdout_Y, 'iloc') else holdout_Y[indices]
            
            # Model 1 predictions and AUC
            prob1 = model1.predict_proba(X_bootstrap1)[:, 1]
            auc1 = roc_auc_score(y_bootstrap, prob1)
            bootstrap_aucs_model1.append(auc1)
            
            # Model 2 predictions and AUC
            prob2 = model2.predict_proba(X_bootstrap2)[:, 1]
            auc2 = roc_auc_score(y_bootstrap, prob2)
            bootstrap_aucs_model2.append(auc2)
        
        bootstrap_aucs_model1 = np.array(bootstrap_aucs_model1)
        bootstrap_aucs_model2 = np.array(bootstrap_aucs_model2)
        
        # Calculate differences (Model1 - Model2)
        differences = bootstrap_aucs_model1 - bootstrap_aucs_model2
        
        # Paired Wilcoxon signed-rank test (two-sided)
        statistic, p_value = wilcoxon(differences, alternative='two-sided')
        
        # Compile results
        fold_comparison = {
            'fold_idx': fold_idx,
            'model1_aucs': bootstrap_aucs_model1,
            'model2_aucs': bootstrap_aucs_model2,
            'differences': differences,
            'model1_mean_auc': np.mean(bootstrap_aucs_model1),
            'model2_mean_auc': np.mean(bootstrap_aucs_model2),
            'mean_difference': np.mean(differences),
            'median_difference': np.median(differences),
            'difference_ci_lower': np.percentile(differences, 2.5),
            'difference_ci_upper': np.percentile(differences, 97.5),
            'wilcoxon_statistic': statistic,
            'p_value': p_value,
            'significant_at_05': p_value < 0.05
        }
        
        comparison_results.append(fold_comparison)
    
    return comparison_results


def print_comparison_summary(comparison_results, model1_name='Model1', model2_name='Model2'):
    """
    Print a summary of the comparison results.
    """
    print(f"\n{'='*80}")
    print(f"Paired Comparison: {model1_name} vs {model2_name}")
    print(f"{'='*80}\n")
    
    for result in comparison_results:
        fold_idx = result['fold_idx']
        print(f"Fold {fold_idx} (Year {fold_idx} → Year {fold_idx+1}):")
        print(f"  {model1_name} AUC: {result['model1_mean_auc']:.4f} "
              f"(95% CI: [{result['model1_aucs'].min():.4f}, {result['model1_aucs'].max():.4f}])")
        print(f"  {model2_name} AUC: {result['model2_mean_auc']:.4f} "
              f"(95% CI: [{result['model2_aucs'].min():.4f}, {result['model2_aucs'].max():.4f}])")
        print(f"  Difference ({model1_name} - {model2_name}): {result['mean_difference']:.4f} "
              f"(95% CI: [{result['difference_ci_lower']:.4f}, {result['difference_ci_upper']:.4f}])")
        print(f"  Wilcoxon p-value: {result['p_value']:.4f} {'***' if result['significant_at_05'] else ''}")
        print()



def plot_delta_auc_with_uncertainty(df, test_years, model1_name='XGBoost', model2_name='EBM', 
                                     title='Model Performance Difference (Δ-AUC) with 95% CI'):

    # Ensure we have the right data
    required_cols = ['mean_difference', 'difference_ci_lower', 'difference_ci_upper']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column: {col}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # X positions
    x_positions = list(range(len(df)))
    
    # Plot the mean difference line
    ax.plot(x_positions, df['mean_difference'], 
            marker='o', linestyle='dashed', linewidth=2, 
            color='#2E86AB', markersize=8,
            label=f'Δ-AUC ({model1_name} - {model2_name})', zorder=3)
    
    # Plot confidence interval as shaded area
    ax.fill_between(x_positions, 
                     df['difference_ci_lower'], 
                     df['difference_ci_upper'],
                     alpha=0.3, color='#2E86AB', 
                     label='95% CI', zorder=2)
    
    # Add horizontal line at y=0 (no difference)
    ax.axhline(y=0, linestyle='-', color='black', linewidth=1, alpha=0.5, zorder=1)
    
    # Set x-tick labels to test years
    x_labels = [str(y) for y in test_years]
    
    # Create custom tick labels, bold for final year
    xtick_labels = []
    for i, label in enumerate(x_labels):
        if i == len(x_labels) - 1:
            label = f"$\\bf{{{label}}}$"  # Render final year in bold
        xtick_labels.append(label)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels)
    
    # Add vertical dashed line before the final test year
    if len(test_years) > 1:
        ax.axvline(x=len(test_years) - 1.5, linestyle='--', color='gray', linewidth=1, zorder=1)
    
    # Labels and title
    ax.set_xlabel('Validate and Test Year')
    ax.set_ylabel(f'Δ-AUC ({model1_name} - {model2_name})')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add text annotation for interpretation
    y_min, y_max = ax.get_ylim()
    if y_max > 0:
        ax.text(0.02, 0.98, f'{model1_name} better →', 
                transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', alpha=0.6)
    if y_min < 0:
        ax.text(0.02, 0.02, f'← {model2_name} better', 
                transform=ax.transAxes, fontsize=9, 
                verticalalignment='bottom', alpha=0.6)
    
    plt.tight_layout()
    plt.show()



def plot_delta_auc_comparison(df_train, df_test, test_years_train, test_years_test,
                               model1_name='XGBoost', model2_name='EBM',
                               title='Model Performance Difference (Δ-AUC) with 95% CI',
                                figsize=(8, 5)):

    fig, ax = plt.subplots(figsize=figsize)
    
    # Concatenate train and test data
    n_train = len(df_train)
    n_test = len(df_test)
    n_total = n_train + n_test
    
    # X positions: train folds first, then test folds
    x_pos_train = list(range(n_train))
    x_pos_test = list(range(n_train, n_total))
    x_positions_all = x_pos_train + x_pos_test
    
    # Combine all difference values and CIs
    mean_diff_all = list(df_train['mean_difference']) + list(df_test['mean_difference'])
    ci_lower_all = list(df_train['difference_ci_lower']) + list(df_test['difference_ci_lower'])
    ci_upper_all = list(df_train['difference_ci_upper']) + list(df_test['difference_ci_upper'])
    
    # Plot continuous line for both train and test
    ax.plot(x_positions_all, mean_diff_all, 
            marker='o', linestyle='dashed', linewidth=2, 
            color='#2E86AB', markersize=8, zorder=3,
            label=f'Δ-AUC ({model1_name} - {model2_name})')
    
    # Plot continuous shaded CI for both train and test
    ax.fill_between(x_positions_all, 
                     ci_lower_all, 
                     ci_upper_all,
                     alpha=0.3, color='#2E86AB', zorder=2,
                     label='95% CI')
    
    # Add horizontal line at y=0 (no difference)
    ax.axhline(y=0, linestyle='-', color='black', linewidth=1, alpha=0.5, zorder=1)
    
    # Add vertical dashed line separating validate and test years
    if n_test > 0:
        ax.axvline(x=n_train - 0.5, linestyle='--', color='gray', linewidth=1, zorder=1)
    
    # Create x-tick labels: validate years (normal), test years (bold)
    all_years = test_years_train + test_years_test
    xtick_labels = []
    for i, year in enumerate(all_years):
        if i >= n_train:  # Test years in bold
            label = f"$\\bf{{{str(year)}}}$"
        else:  # Validate years normal
            label = str(year)
        xtick_labels.append(label)
    
    ax.set_xticks(x_positions_all)
    ax.set_xticklabels(xtick_labels)
    
    # Labels and title
    ax.set_xlabel('Validate and Test Year')
    ax.set_ylabel(f'Δ-AUC ({model1_name} - {model2_name})')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

