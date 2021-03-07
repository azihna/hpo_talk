# %%
import time

from sklearn import datasets, metrics
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

data = datasets.fetch_california_housing()
X, y = data['data'], data['target']

# %%
metric = 'neg_mean_squared_error'
num_cv_folds = 5
est = XGBRegressor()

params = {'n_estimators': [150, 200, 250], 
          'max_depth': [4, 5, 6]}

hpo = GridSearchCV(est,
                   params,
                   scoring=metric,
                   cv=num_cv_folds)

t0 = time.time()
hpo.fit(X, y)
print(f'elapsed: {time.time() - t0:.4}')
print(f'best results: {hpo.best_score_}')
