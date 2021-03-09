import numpy as np
from sklearn import datasets, metrics, model_selection
from hyperopt import hp, fmin, tpe, Trials
from xgboost import XGBRegressor

# %%
data = datasets.fetch_california_housing()
X, y = data['data'], data['target']

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
    

params = {'n_estimators': hp.randint('n_estimators', 200, 500), 
          'learning_rate': hp.loguniform('learning_rate', 
                                         np.log(0.001), 
                                         np.log(0.1)),
          'max_depth': hp.randint('max_depth', 4, 7), 
          'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart'])}

trials = Trials()

hpo = fmin(fn=objective, 
           space=params,
           algo=tpe.suggest,
           max_evals=10,
           trials=trials)

print(f'best run: {hpo}')
