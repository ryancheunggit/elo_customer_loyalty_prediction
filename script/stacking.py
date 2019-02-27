import gc; gc.enable()
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from feature import get_data, get_folds
from model import rmse

target = pd.read_csv('../data/train.csv')['target']
b_target = target.map(lambda x: 1 if x < -33 else 0)

train_metas = [
    '../oof/regression_model_lgb_data_mine_date_2-24-0-43.npy',
    '../oof/regression_model_lgb_data_mine_shift_-1_date_2-24-1-48.npy',
    '../oof/regression_model_lgb_data_mine_shift_-2_date_2-24-2-50.npy',
    '../oof/regression_model_lgb_data_mine_shift_-3_date_2-24-3-49.npy',
    '../oof/regression_model_lgb_data_mine_shift_-4_date_2-24-4-41.npy',
    '../oof/regression_model_xgb_data_mine_date_2-24-12-58.npy',
    '../oof/regression_model_xgb_data_mine_shift_-1_date_2-24-12-59.npy',
    '../oof/regression_model_xgb_data_mine_shift_-2_date_2-24-13-0.npy',
    '../oof/regression_model_xgb_data_mine_shift_-3_date_2-24-13-1.npy',
    '../oof/regression_model_xgb_data_mine_shift_-4_date_2-24-13-2.npy',
    '../oof/regression_model_rgf_data_mine_date_2-24-21-58.npy',
    '../oof/classification_model_lgb_data_mine_date_2-25-10-28.npy',
    '../oof/classification_model_xgb_data_mine_date_2-25-10-38.npy',
]

test_metas = [
    '../submission/regression_model_lgb_data_mine_date_2-24-0-43.csv',
    '../submission/regression_model_lgb_data_mine_shift_-1_date_2-24-1-48.csv',
    '../submission/regression_model_lgb_data_mine_shift_-2_date_2-24-2-50.csv',
    '../submission/regression_model_lgb_data_mine_shift_-3_date_2-24-3-49.csv',
    '../submission/regression_model_lgb_data_mine_shift_-4_date_2-24-4-41.csv',
    '../submission/regression_model_xgb_data_mine_date_2-24-12-58.csv',
    '../submission/regression_model_xgb_data_mine_shift_-1_date_2-24-12-59.csv',
    '../submission/regression_model_xgb_data_mine_shift_-2_date_2-24-13-0.csv',
    '../submission/regression_model_xgb_data_mine_shift_-3_date_2-24-13-1.csv',
    '../submission/regression_model_xgb_data_mine_shift_-4_date_2-24-13-2.csv',
    '../submission/regression_model_rgf_data_mine_date_2-24-21-58.csv',
    '../submission/classification_model_lgb_data_mine_date_2-25-10-28.csv',
    '../submission/classification_model_xgb_data_mine_date_2-25-10-38.csv',
]

train_meta = pd.concat([pd.Series(np.load(path)) for path in train_metas], axis=1)
test_meta  = pd.concat([pd.Series(pd.read_csv(path)['target']) for path in test_metas], axis=1)
train_meta.columns = [
    'lgb_lag_0',
    'lgb_lag_1',
    'lgb_lag_2',
    'lgb_lag_3',
    'lgb_lag_4',
    'xgb_lag_0',
    'xgb_lag_1',
    'xgb_lag_2',
    'xgb_lag_3',
    'xgb_lag_4',
    'rgf_lag_0',
    'lgb_clf_lag_0',
    'xgb_clf_lag_0',
    ]

test_meta.columns = train_meta.columns
folds = get_folds(n_folds=5, seed=42, verbose=0, recompute=0)
sklearn_rmse = make_scorer(rmse, greater_is_better=False)

cv = GridSearchCV(
    estimator=Ridge(),
    param_grid={'alpha': [500, 800, 1000, 1200, 1500, 2000, 3000, 4000, 5000, 8000, 10000]},
    scoring=sklearn_rmse,
    n_jobs=1,
    cv=folds,
    verbose=1
)
cv.fit(train_meta, target)
print(pd.DataFrame(cv.cv_results_)[['params', 'mean_test_score']])

reg_model = Ridge(alpha=5000).fit(train_meta, target)
print(reg_model.coef_)

test_pred = reg_model.predict(test_meta)
sub.target = test_pred
sub.to_csv('../submission/ridge_stack_2.csv', index=False)
