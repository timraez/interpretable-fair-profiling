import pandas as pd
from scripts.fit_evaluate_models import fit_lr, get_results_lr
import matplotlib.pyplot as plt

def fit_lr_parameter_range(folds, dataset_info, parameter_range):
    weights_parameters = {}
    scores_parameters = {}
    for parameter in parameter_range:
        print('Calculating with parameter '+str(parameter)+' ...')
        folds_models = fit_lr(folds, dataset_info, parameter)
        weights_folds, scores_folds = get_results_lr(folds_models, dataset_info)
        weights_pd = pd.DataFrame(weights_folds)
        weights_parameters[str(parameter)] = weights_pd
        scores_parameters[str(parameter)] = scores_folds
        print('Scores for parameter '+str(parameter)+': '+str(scores_folds))
    scores_pd = pd.DataFrame(scores_parameters)
    return weights_parameters, scores_pd


def get_number_nonzero_weights(weights_parameters):
    no_important = []
    for weights in weights_parameters.values():
        weight_average = weights.sum().fillna(0)
        important = (weight_average != 0).sum()
        no_important.append(important)
    return no_important


import matplotlib.pyplot as plt

def plot_dual_axis_nonzero(parameter_range, scores, nonzero_weights):
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot AUC on the left y-axis
    auc_line, = ax1.plot(scores, color='tab:blue', marker='o', label='AUC')
    ax1.set_xlabel('L1 parameter')
    ax1.set_ylabel('Average AUC score', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(range(len(parameter_range)))
    ax1.set_xticklabels(parameter_range, rotation=90)

    # Create a second y-axis for number of non-zero weights
    ax2 = ax1.twinx()
    nonzero_line, = ax2.plot(nonzero_weights, color='tab:green', marker='s', label='Non-Zero Weights')
    ax2.set_ylabel('Number of non-zero weights', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Add grid and title
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    plt.title('LR: AUC and Feature Sparsity vs. L1 Regularization')

    # Combined legend from both axes
    lines = [auc_line, nonzero_line]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='lower center')

    fig.tight_layout()
    plt.show()

