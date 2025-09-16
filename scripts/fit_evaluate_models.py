from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import copy
import pandas as pd
import itertools
import joblib
import matplotlib.pyplot as plt

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

