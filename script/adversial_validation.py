import pandas as pd
import numpy as np
from model import *
from feature import *
from sklearn.model_selection import StratifiedKFold
train, test, target, sub, feature_cols = get_data()
params = get_params(model_name='lgb', task='classification')
train['dataset'] = 0
test['dataset'] = 1
data = pd.concat([train, test], axis=0)
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(data, data['dataset']))

cv_results = lgb_cv(X=data[feature_cols], y=data['dataset'], X_test=data[feature_cols], folds=folds, params=params, predict=False, verbose=True)
