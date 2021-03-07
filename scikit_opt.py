#%%
import time

import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_evaluations, plot_gaussian_process
from xgboost import XGBRegressor

data = datasets.fetch_california_housing()
X, y = data['data'], data['target']

# %%
metric = 'neg_mean_squared_error'
num_cv_folds = 5
est = XGBRegressor()

params = {'n_estimators': Integer(200, 500), 
          'learning_rate': Real(0.001, 0.1, prior='log-uniform'),
          'max_depth': Integer(4, 7), 
          'booster': Categorical(['gbtree', 'gblinear', 'dart'])}

hpo = BayesSearchCV(est,
                   params,
                   scoring=metric,
                   cv=num_cv_folds,
                   verbose=1,
                   n_jobs=-1,
                   n_iter=10)

t0 = time.time()
hpo.fit(X, y)
print(f'elapsed: {time.time() - t0:.4}')
print(f'best results: {hpo.best_score_}')

# %%
_ = plot_evaluations(hpo, bins=10)

# %%
_ = plot_objective(hpo.optimizer_results_[0],
                   dimensions=['n_estimators', 'learning_rate',
                               'max_depth', 'booster'],
                   n_minimum_search=int(1e8))
plt.show()
# %%

plot_gaussian_process(hpo, params)
# %%
