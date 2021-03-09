# %%
import numpy as np
import ray
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch
from sklearn import datasets, metrics, model_selection
from xgboost import XGBRegressor

# %%
data = datasets.fetch_california_housing()
X, y = data['data'], data['target']

# %%
num_cv_folds = 5

skfolds = model_selection.KFold(n_splits=num_cv_folds)


def objective(config):
    for fold, (train_index, test_index) in enumerate(skfolds.split(X, y)):
        est = XGBRegressor(**config)
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        est.fit(X_tr, y_tr)
        preds = est.predict(X_te)
        score = metrics.mean_squared_error(y_te, preds)
        tune.report(iterations=fold, mean_loss=score)


ray.init(configure_logging=False)

params = {}
params['n_estimators'] = tune.randint(200, 500)
params['learning_rate'] = tune.loguniform(0.001, 0.1)
params['booster'] = tune.choice(['gbtree', 'gblinear', 'dart'])
params['max_depth'] = tune.randint(4, 7)

algo = OptunaSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=4)

hpo = tune.run(
    objective,
    metric='mean_loss',
    mode='min',
    search_alg=algo,
    num_samples=10,
    config=params
)

print(f'best hyperparameters are: {hpo.best_config}')
# %%
df = hpo.results_df
print(df.head())
