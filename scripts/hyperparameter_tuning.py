import optuna
import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingClassifier

def xgb_objective(trial, folds):
    # Define hyperparameters to tune
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,  # suppress warning in older XGBoost versions
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1)
    }

    auc_scores = []

    for fold in folds:
        CV_X = fold['CV_X']
        CV_Y = fold['CV_Y']

        holdout_X = fold['holdout_X'].copy()
        holdout_Y = fold['holdout_Y']

        clf = xgb.XGBClassifier(**param)
        clf.fit(CV_X, CV_Y)

        probabilities = clf.predict_proba(holdout_X)[:, 1]
        auc_scores.append(roc_auc_score(holdout_Y, probabilities))

    return np.mean(auc_scores)

def run_xgb_hyperparameter_tuning(folds, n_trials=50):
    def objective_with_data(trial):
        return xgb_objective(trial, folds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_data, n_trials=n_trials)

    print(f'Best hyperparameters: {study.best_params}')
    print(f'Best AUC: {study.best_value}')
    
    return study

def objective_EBM(trial, folds, use_all_folds=True):
    param = {
        'interactions': trial.suggest_int('interactions', 30, 80),
        'outer_bags': trial.suggest_int('outer_bags', 8, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 5),
        'max_leaves': trial.suggest_int('max_leaves', 2, 3),
    }

    selected_folds = folds if use_all_folds else [folds[0]]
    auc_scores = []

    for fold in selected_folds:
        CV_X = fold['CV_X']
        CV_Y = fold['CV_Y']

        clf = ExplainableBoostingClassifier(**param).fit(CV_X, CV_Y)

        holdout_X = fold['holdout_X']
        holdout_Y = fold['holdout_Y']

        probabilities = clf.predict_proba(holdout_X)[:, 1]
        auc_score = roc_auc_score(holdout_Y, probabilities)
        auc_scores.append(auc_score)

    return np.mean(auc_scores)

def run_ebm_hyperparameter_tuning(folds, n_trials=20, initial_params=None, use_all_folds=True):
    study = optuna.create_study(direction='maximize')

    if initial_params is not None:
        study.enqueue_trial(initial_params)

    study.optimize(lambda trial: objective_EBM(trial, folds, use_all_folds=use_all_folds), n_trials=n_trials)

    print(f'Best hyperparameters: {study.best_params}')
    print(f'Best AUC: {study.best_value}')
    return study

