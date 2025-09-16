# Transparent and Fair Profiling in Employment Services: Evidence from Switzerland

This is the code for the paper "Transparent and Fair Profiling in Employment Services: Evidence from Switzerland" by Tim Räz. It provides a proof-of-concept that interpretable and fair statistical profiling of long-term unemployment is feasible. It employs real administrative data from Switzerland. Two interpretable models (logistic regression, explainable boosting machines) as well as three black-box models (random forest, gradient boosting, extreme gradient boosting) are fit and evaluated. EBM is only slightly worse than GB and XGB, but globally interpretable. Fairness mitigation is also performed. Note that the data used in this project is not publicly available due to privacy restrictions; see the paper for instructions of how to obtain the data.

To replicate findings, run the five Jupyter notebooks as needed:

  1. Run main1_merge_preprocess to preprocess data.
  2. Run main2_fit_evaluate_models to replicate predictive performance results on both train-validate and test sets.
  3. Run main3_hyperparameter_tuning for hyperparameter tuning (optional).
  4. Run main4_interpretability to replicate interpretability results, including sparsity and smoothing feature functions.
  5. Run main5_fairness to replicate fairness results.

Code was run and tested with Python 3.11.9. and Jupyter notebooks. See requirements.txt for used Python packages and versions.
