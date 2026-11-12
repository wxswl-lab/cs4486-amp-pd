# CS4486 AMP-PD

Patient-disjoint five-fold evaluation of five longitudinal UPDRS forecasting models.

1. Download the training data from [Kaggle](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/data).
2. Install NumPy and pandas with pip install -r requirements.txt.
3. Run python run_experiments.py PATH_TO_DATA --out results.

The script rebuilds the folds with seed 4486 and exports cv_summary.csv and cv_breakdown.csv. This repository contains code and aggregate results.
