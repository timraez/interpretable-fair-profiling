import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from interpret.glassbox import ExplainableBoostingClassifier
import copy

def fit_evaluate_ebms_l1(alphas, fold, fixed_params):
    results = []
    retained_main_features = {}

    CV_X, CV_Y = fold['CV_X'].copy(), fold['CV_Y'].copy()
    holdout_X, holdout_Y = fold['holdout_X'].copy(), fold['holdout_Y'].copy()

    for i, alpha in enumerate(alphas, 1):
        print(f"[{i}/{len(alphas)}] Fitting with reg_alpha = {alpha}")

        params = fixed_params.copy()
        params['reg_alpha'] = alpha
        model = ExplainableBoostingClassifier(**params)
        model.fit(CV_X, CV_Y)

        preds = model.predict_proba(holdout_X)[:, 1]
        auc = roc_auc_score(holdout_Y, preds)

        # Count non-zero terms
        nonzero_terms = sum(np.any(np.abs(term) > 0) for term in model.term_scores_)

        # Extract main feature names with non-zero effect
        main_features = []
        for term_scores, term_feat in zip(model.term_scores_, model.term_features_):
            if len(term_feat) == 1 and np.any(np.abs(term_scores) > 0):
                feature_idx = term_feat[0]
                feature_name = model.feature_names_in_[feature_idx]
                main_features.append(feature_name)

        retained_main_features[alpha] = main_features

        results.append({
            'reg_alpha': alpha,
            'auc': auc,
            'n_nonzero_terms': nonzero_terms,
            'n_retained_main_features': len(main_features)
        })

    df = pd.DataFrame(results)
    return df, retained_main_features


def plot_ebm_results_l1(df_results, label_rotate=True, step=1):
    df_results = df_results.sort_values('reg_alpha').reset_index(drop=True)
    indices = np.arange(len(df_results))

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot AUC
    auc_line, = ax1.plot(indices, df_results['auc'], color='tab:blue', marker='o', label='AUC')
    ax1.set_xlabel('Regularization Strength (reg_alpha)')
    ax1.set_ylabel('AUC score', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, axis='x', linestyle='--', linewidth=0.5)

    # Setup x-ticks and labels
    ticks = indices[::step]
    labels = [f"{val:g}" for val in df_results['reg_alpha'].iloc[::step]]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels, rotation='vertical' if label_rotate else 'horizontal')

    # Plot non-zero terms
    ax2 = ax1.twinx()
    nonzero_line, = ax2.plot(indices, df_results['n_nonzero_terms'], color='tab:red', marker='s', label='Non-Zero Terms')
    ax2.set_ylabel('Number of Non-Zero Terms', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add combined legend
    lines = [auc_line, nonzero_line]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='lower center')

    plt.title('EBM: AUC and Number of Non-Zero Terms vs. L1 Regularization')
    fig.tight_layout()
    plt.show()


def count_main_effects(ebm):
    # Each entry in ebm.term_features_ is a tuple of feature indices.
    # Main effects correspond to tuples of length 1.
    return sum(len(f_idxs) == 1 for f_idxs in ebm.term_features_)


def evaluate_top_n_features(fold, top_features_list, params):
    # 1. Extract feature importances from the trained EBM
    ebm = fold['model']
    ebm_global = ebm.explain_global()
    fi = ebm_global.data()
    feature_names = fi['names']
    feature_importances = fi['scores']

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # 2. Keep main effects only
    main_effects_df = importance_df[
        ~importance_df['Feature'].str.contains(' & ')
    ].reset_index(drop=True)

    # 3. Loop over requested top-Ns and evaluate
    X_train = fold['CV_X']
    X_test = fold['holdout_X']
    y_train = fold['CV_Y']
    y_test = fold['holdout_Y']

    auc_sparsities = {}

    for top_n in top_features_list:
        selected = main_effects_df['Feature'].head(top_n).tolist()
        Xtr = X_train[selected]
        Xte = X_test[selected]

        ebm_sparse = ExplainableBoostingClassifier(**params)
        ebm_sparse.fit(Xtr, y_train)

        y_pred = ebm_sparse.predict_proba(Xte)[:, 1]
        auc_val = roc_auc_score(y_test, y_pred)
        auc_sparsities[top_n] = auc_val

        print(f"Top {top_n} features → AUC: {auc_val:.4f}")

    return auc_sparsities, main_effects_df


def get_sparse_ebm(folds, sparsity):
    auc_folds = []
    new_folds = []

    for fold in folds:
        new_fold = copy.deepcopy(fold)

        model_original = new_fold['model']
        ebm_global = model_original.explain_global()
        fi = ebm_global.data()
        importance_df = pd.DataFrame({
            'Feature': fi['names'],
            'Importance': fi['scores']
        }).sort_values(by='Importance', ascending=False)

        main_effects_df = importance_df[~importance_df['Feature'].str.contains(' & ')]

        X_train = new_fold['CV_X']
        X_test = new_fold['holdout_X']
        y_train = new_fold['CV_Y']
        y_test = new_fold['holdout_Y'].to_numpy().copy()

        selected = main_effects_df['Feature'].head(sparsity).tolist()
        Xtr = X_train[selected]
        Xte = X_test[selected]

        model_sparse = ExplainableBoostingClassifier(interactions=0.5)
        model_sparse.fit(Xtr, y_train)

        preds = model_sparse.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y_test, preds)
        auc_folds.append(auc)

        new_fold['model_sparse'] = model_sparse
        new_folds.append(new_fold)

    return new_folds, auc_folds


def plot_auc_sparse(auc_sparse):

    plt.figure(figsize=(8, 6))
    plt.plot(auc_sparse, marker='o', label='AUC')

    # Axis labels and title
    plt.xlabel("Number of main features")
    plt.ylabel("AUC Score")
    plt.title("EBM: AUC for Given Number of Main Features (Sparsity)")

    # Enable grid for easier value reading
    plt.grid(True, linestyle='--', linewidth=0.5)  # gridlines on both axes :contentReference[oaicite:2]{index=2}

    plt.tight_layout()
    plt.show()

def get_ebm_functions(model, feature_names):
    functions = {}
    for feature in feature_names:
        term_index = model.term_names_.index(feature)
        y_values = model.term_scores_[term_index]
        functions[feature] = {
            'raw': y_values,
            'smooth': None  # to be filled later
        }
    return functions

from scipy.interpolate import make_smoothing_spline


def apply_smoothing_spline(functions_dict, function_label, parameter, plot=True):

    raw = np.asarray(functions_dict[function_label]['raw'])
    n = len(raw)
    if n < 3:
        raise ValueError("Function must have at least 3 points to smooth the interior.")

    x = np.arange(n)
    x_inner = x[1:-1]
    y_inner = raw[1:-1]

    # Fit only on interior points
    spline = make_smoothing_spline(x_inner, y=y_inner, lam=parameter)

    # Construct full smoothed array, preserving endpoints
    smooth = raw.copy()
    smooth[1:-1] = spline(x_inner)

    functions_dict[function_label]['smooth'] = smooth

    # Plot the raw and smoothed functions
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(x_inner, y_inner, 'r', label='Raw')
        plt.plot(x_inner, spline(x_inner), 'b', label='Smoothed')
        #plt.scatter([0, n - 1], raw[[0, n - 1]], color='k', label='Endpoints (unchanged)')
        plt.xlabel(f"{function_label} index")
        plt.ylabel("Score")
        plt.title(f"Smoothing Spline of '{function_label}' (λ={parameter})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    return functions_dict


def replace_with_smooth_terms(original_model, smoothed_functions_dict):

    model_copy = copy.deepcopy(original_model)

    for feature, functions in smoothed_functions_dict.items():
        if feature not in model_copy.term_names_:
            raise ValueError(f"Feature '{feature}' not found in model.term_names_.")

        term_index = model_copy.term_names_.index(feature)

        # Check shape match (optional but good practice)
        original_shape = model_copy.term_scores_[term_index].shape
        new_shape = np.asarray(functions["smooth"]).shape

        if original_shape != new_shape:
            raise ValueError(f"Shape mismatch for feature '{feature}': "
                             f"expected {original_shape}, got {new_shape}")

        # Replace the term score
        model_copy.term_scores_[term_index] = np.asarray(functions["smooth"])

    return model_copy

import copy
import numpy as np
from sklearn.metrics import roc_auc_score

def get_smooth_ebm(folds, feature_names, smoothing_parameters):

    new_folds = []
    auc_scores = []

    for fold in folds:
        f = copy.deepcopy(fold)

        model_sparse = f.get('model_sparse')
        if model_sparse is None:
            raise KeyError("Each fold must contain 'model_sparse'")

        # Extract EBM functions to smooth
        functions_dict = get_ebm_functions(model_sparse, feature_names)

        # Apply smoothing per feature (no plot)
        for feature in feature_names:
            lam = smoothing_parameters.get(feature)
            if lam is None:
                raise KeyError(f"Missing smoothing parameter for feature '{feature}'")
            functions_dict = apply_smoothing_spline(
                functions_dict, feature, lam, plot=False
            )

        # Create the smoothed EBM
        model_smooth = replace_with_smooth_terms(model_sparse, functions_dict)
        f['model_sparse_smooth'] = model_smooth

        # Evaluate on hold-out
        X_ho = f.get('holdout_X')
        y_ho = f.get('holdout_Y')
        if X_ho is None or y_ho is None:
            raise KeyError("Each fold must contain 'holdout_X' and 'holdout_Y'")

        y_pred = model_smooth.predict_proba(X_ho)[:, 1]
        auc = roc_auc_score(y_ho, y_pred)
        auc_scores.append(auc)

        new_folds.append(f)

    return new_folds, auc_scores
