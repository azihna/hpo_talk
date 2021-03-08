# %%
import numpy as np
from sklearn import datasets, metrics, model_selection
import optuna
from xgboost import XGBRegressor

# %%
data = datasets.fetch_california_housing()
X, y = data['data'], data['target']

# %%
num_cv_folds = 5

skfolds = model_selection.KFold(n_splits=num_cv_folds)


def objective(params):
    scores = []
    for train_index, test_index in skfolds.split(X, y):
        est = XGBRegressor(**params)
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        est.fit(X_tr, y_tr)
        preds = est.predict(X_te)
        score = metrics.mean_squared_error(y_te, preds)
        scores.append(score)
    return np.mean(scores)


def optimize(trial):
    params = {}
    params['n_estimators'] = trial.suggest_int('n_estimator', 200, 500)
    params['learning_rate'] = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
    params['booster'] = trial.suggest_categorical('booster', 
                                                  ['gbtree', 'gblinear', 'dart'])
    if params['booster'] != 'gblinear':
        params['max_depth'] = trial.suggest_int('max_depth', 4, 7)
    
    return objective(params)

study = optuna.create_study(
    direction='minimize'
)

study.optimize(optimize, n_trials=10)

print(study.best_params)

# %%
optuna.visualization.plot_param_importances(study)

# %%
optuna.visualization.plot_slice(study)

# %%
optuna.visualization.plot_contour(study)