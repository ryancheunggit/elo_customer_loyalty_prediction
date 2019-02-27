import gc; gc.enable()
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pprint import pprint
from rgf.sklearn import FastRGFRegressor, FastRGFClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score


def rmse(y_true, y_pred): return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


def get_params(model_name='lgb', task='regression'):
    if model_name == 'lgb':
        params = {
            'objective': 'regression_l2',
            'boosting': 'gbdt',
            'learning_rate': 0.005,
            'num_leaves': 127,
            'max_depth': 9,
            'min_data_in_leaf': 150,
            'bagging_freq': 1,
            'bagging_fraction': 0.88,
            'feature_fraction': 0.88,
            'lambda_l1': 10,
            'lambda_l2': 0.5,
            'metric': 'rmse',
            'verbose':-1
        }
        if task == 'classification':
            params.update({
                'objective': 'binary',
                'metric': 'auc',
            })
        if task == 'rank':
            params.update({
                'objective': 'lambdarank',
                'metric': 'ndcg'
            })
    if model_name == 'xgb':
        params = {
            'objective': 'reg:linear',
            'booster': 'gbtree',
            'tree_method': 'gpu_hist',
            'learning_rate': 0.01,
            'max_depth': 7,
            'min_child_weight': 50,
            'subsample': 0.67,
            'colsample_bytree': 0.76,
            'gamma': 10.0,
            'reg_alpha': 0,
            'reg_lambda': 20,
            'eval_metric': 'rmse',
            'verbosity': 0
        }
        if task == 'classification':
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc'
            })
        if task == 'rank':
            params.update({
                'objective': 'rank:pairwise',
                'eval_metric': 'ndcg'
            })
    if model_name == 'rfc':
        params = {
            'n_estimators': 3000,
            'n_jobs': -1,
            'verbose': 0
        }
    if model_name == 'rfr':
        params = {
            'n_estimators': 3000,
            'n_jobs': -1,
            'verbose': 0
        }
    if model_name == 'rgf':
        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'max_leaf': 16,
            'l1': 1000.0,
            'l2': 8.0,
            'tree_gain_ratio': 0.49,
            'opt_algorithm': 'rgf',
            'max_bin': 10
        }
    return params


def make_group(length, n_groups=5):
    group_size = length // n_groups
    residual = length - group_size * n_groups
    groups = np.ones(n_groups) * group_size
    groups[:residual] += 1
    return groups.astype(int)


def lgb_cv(X, y, X_test, folds, params, group_size=1000, predict=False, verbose=False):
    obj = params['objective']
    metric = 'ndcg@5' if obj == 'lambdarank' else params['metric']

    lgb_models = []
    cv_scores = []
    num_iterations = []
    fis = []
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
    params['min_data_in_leaf'] = int(params['min_data_in_leaf'])

    if verbose: pprint(params)
    verbose_eval = 100 if verbose else False

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        if verbose: print('+ fitting lgb on fold {}'.format(fold_idx + 1))
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        if obj == 'lambdarank':
            lgb_train = lgb.Dataset(data=X_train, label=y_train, group=make_group(len(train_idx), 40))
            lgb_valid = lgb.Dataset(data=X_valid, label=y_valid, group=make_group(len(valid_idx), 10), reference=lgb_train)
        else:
            lgb_train = lgb.Dataset(data=X_train, label=y_train)
            lgb_valid = lgb.Dataset(data=X_valid, label=y_valid, reference=lgb_train)

        model = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=65535,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            early_stopping_rounds=100,
            verbose_eval= verbose_eval
        )
        if obj == 'regression' and verbose:
            print(rmse(y.iloc[valid_idx], model.predict(X.iloc[valid_idx])))
        lgb_models.append(model)
        cv_scores.append(model.best_score['valid'][metric])
        num_iterations.append(model.best_iteration)
        fis.append({k:v for k, v in zip(model.feature_name(), model.feature_importance())})
        if predict:
            oof_preds[valid_idx] = model.predict(data=X_valid)
            test_preds += model.predict(data=X_test) / len(folds)
        del X_train, y_train, X_valid, y_valid, lgb_train, lgb_valid
        gc.collect()

    fis = pd.DataFrame(fis).T
    fis['mean_feature_importance'] = fis.mean(axis=1)
    fis.sort_values('mean_feature_importance', ascending=False, axis=0, inplace=True)
    fis = fis.reset_index()
    return {
        'trained_model': lgb_models,
        'cv_scores': cv_scores,
        'num_iterations': num_iterations,
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'fis': fis
    }


def lgb_cv_opt(X, y, X_test, folds, max_depth, num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, lambda_l1, lambda_l2):
    params = {
        'objective': 'regression_l2',
        'boosting': 'gbdt',
        'learning_rate': 0.005,
        'num_leaves': int(num_leaves),
        'max_depth': int(max_depth),
        'min_data_in_leaf': int(min_data_in_leaf),
        "bagging_freq": 1,
        "bagging_fraction": bagging_fraction ,
        "feature_fraction": feature_fraction,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        'metric': 'rmse',
        "verbosity": -1
    }
    if pd.Series(y).nunique() == 2:
        params.update({'objective': 'binary', 'metric': 'auc'})
        results = lgb_cv(X, y, X_test, folds, params, predict=False, verbose=False)
        mean_cv_score = np.mean(results['cv_scores'])
        return mean_cv_score
    else:
        results = lgb_cv(X, y, X_test, folds, params, predict=False, verbose=False)
        mean_cv_score = np.mean(results['cv_scores'])
        return -mean_cv_score


def xgb_cv(X, y, X_test, folds, params, predict=False, verbose=False):
    xgb_models = []
    cv_scores = []
    num_iterations = []
    fis = []
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    xgb_test = xgb.DMatrix(data=X_test)

    params['max_depth'] = int(params['max_depth'])

    if verbose: pprint(params)
    verbose_eval = 100 if verbose else False

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        if verbose: print('+ fitting xgb on fold {}'.format(fold_idx + 1))
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        xgb_train = xgb.DMatrix(data=X_train, label=y_train)
        xgb_valid = xgb.DMatrix(data=X_valid, label=y_valid)

        model = xgb.train(
            params=params,
            dtrain=xgb_train,
            num_boost_round=65535,
            evals=[(xgb_train, 'train'), (xgb_valid, 'valid')],
            early_stopping_rounds=100,
            verbose_eval= verbose_eval
        )

        xgb_models.append(model)
        cv_scores.append(model.best_score)
        num_iterations.append(model.best_ntree_limit)
        fis.append(model.get_fscore())
        if predict:
            oof_preds[valid_idx] = model.predict(data=xgb_valid, ntree_limit=model.best_ntree_limit)
            test_preds += model.predict(data=xgb_test, ntree_limit=model.best_ntree_limit) / len(folds)
        del X_train, y_train, X_valid, y_valid, xgb_train, xgb_valid
        gc.collect()

    fis = pd.DataFrame(fis).T
    fis['mean_feature_importance'] = fis.mean(axis=1)
    fis.sort_values('mean_feature_importance', ascending=False, axis=0, inplace=True)
    fis = fis.reset_index()
    return {
        'trained_model': xgb_models,
        'cv_scores': cv_scores,
        'num_iterations': num_iterations,
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'fis': fis
    }


def xgb_cv_opt(X, y, X_test, folds, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1, gamma=0, reg_alpha=0, reg_lambda=1):
    params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'tree_method': 'gpu_hist',
        'eval_metric': 'rmse',
        'learning_rate': 0.005,
        'max_depth': int(max_depth),
        'min_child_weight': min_child_weight,
        'subsample': 1,
        'colsample_bytree': 1,
        'gamma': gamma,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'verbosity': 0
    }
    if pd.Series(y).nunique() == 2:
        params.update({'objective': 'binary:logistic', 'eval_metric': 'auc'})
        results = xgb_cv(X, y, X_test, folds, params, predict=False, verbose=False)
        mean_cv_score = np.mean(results['cv_scores'])
        return mean_cv_score
    else:
        results = xgb_cv(X, y, X_test, folds, params, predict=False, verbose=False)
        mean_cv_score = np.mean(results['cv_scores'])
        return -mean_cv_score


def rfc_cv(X, y, X_test, folds, params, predict, verbose=False):
    rfc_models = []
    cv_scores = []
    oof_probas = np.zeros(X.shape[0])
    test_probas = np.zeros(X_test.shape[0])
    fis = []

    X.replace(np.inf, 1 / 1e-15, inplace=True)
    X_test.replace(np.inf, 1 / 1e-15, inplace=True)

    if verbose: pprint(params)
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        if verbose: print('+ fitting rfc on fold {}'.format(fold_idx + 1))
        X_train, y_train = X.iloc[train_idx].astype('float32'), y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx].astype('float32'), y.iloc[valid_idx]

        model = RandomForestClassifier(**params).fit(X=X_train, y=y_train)
        rfc_models.append(model)
        cv_scores.append(roc_auc_score(y_valid, model.predict_proba(X_valid)[:,1]))
        if predict:
            oof_probas[valid_idx] = model.predict_proba(X_valid)[:,1]
            test_probas += model.predict_proba(X_test)[:,1] / len(folds)
        fis.append(model.feature_importances_)
        del X_train, y_train, X_valid, y_valid
        gc.collect()

    fis = pd.DataFrame(fis)
    fis.columns = X.columns
    fis = fis.T
    fis['mean_feature_importance'] = fis.mean(axis=1)
    fis.sort_values('mean_feature_importance', ascending=False, axis=0, inplace=True)
    fis = fis.reset_index()

    return {
        'trained_model': rfc_models,
        'cv_scores': cv_scores,
        'oof_preds': oof_probas,
        'test_preds': test_probas,
        'fis': fis
    }


def rfr_cv(X, y, X_test, folds, params, predict, verbose=False):
    rfr_models = []
    cv_scores = []
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    fis = []

    X.replace(np.inf, 1 / 1e-15, inplace=True)
    X_test.replace(np.inf, 1 / 1e-15, inplace=True)

    if verbose: pprint(params)
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        if verbose: print('+ fitting rfc on fold {}'.format(fold_idx + 1))
        X_train, y_train = X.iloc[train_idx].astype('float32'), y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx].astype('float32'), y.iloc[valid_idx]

        model = RandomForestRegressor(**params).fit(X=X_train, y=y_train)
        rfr_models.append(model)
        valid_pred = model.predict(X_valid)
        cv_scores.append(rmse(y_valid, valid_pred))
        if predict:
            oof_preds[valid_idx] = valid_pred
            test_preds += model.predict(X_test) / len(folds)
        fis.append(model.feature_importances_)
        del X_train, y_train, X_valid, y_valid
        gc.collect()

    fis = pd.DataFrame(fis)
    fis.columns = X.columns
    fis = fis.T
    fis['mean_feature_importance'] = fis.mean(axis=1)
    fis.sort_values('mean_feature_importance', ascending=False, axis=0, inplace=True)
    fis = fis.reset_index()

    return {
        'trained_model': rfr_models,
        'cv_scores': cv_scores,
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'fis': fis
    }


def rfr_cv_opt(X, y, X_test, folds, n_estimators=3000):
    params = {'n_estimators': int(n_estimators)}
    results = rfr_cv(X, y, X_test, folds, params, predict=False, verbose=False)
    mean_cv_score = np.mean(results['cv_scores'])
    return -mean_cv_score


def rgf_cv(X, y, X_test, folds, params, predict, verbose=False):
    if pd.Series(y).nunique() == 2:
        metric = roc_auc_score
        rgf_model = FastRGFClassifier
    else:
        metric = rmse
        rgf_model = FastRGFRegressor
    rgf_models = []
    cv_scores = []
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['max_leaf'] = int(params['max_leaf'])
    params['tree_gain_ratio'] = float(params['tree_gain_ratio'])
    params['l1'] = float(params['l1'])
    params['l2'] = float(params['l2'])

    if verbose: pprint(params)
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        if verbose: print('+ fitting rgf on fold {}'.format(fold_idx + 1))
        X_train, y_train = X.iloc[train_idx].astype('float32'), y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx].astype('float32'), y.iloc[valid_idx]

        model = rgf_model(**params).fit(X=X_train, y=y_train)
        rgf_models.append(model)
        val_pred = model.predict(X_valid)
        valid_metric_score = metric(y_valid, val_pred)
        if verbose: print(valid_metric_score)
        cv_scores.append(valid_metric_score)
        if predict:
            oof_preds[valid_idx] = val_pred
            test_preds += model.predict(X_test) / len(folds)
        del X_train, y_train, X_valid, y_valid
        gc.collect()

    return {
        'trained_model': rgf_models,
        'cv_scores': cv_scores,
        'num_iterations': params['n_estimators'],
        'oof_preds': oof_preds,
        'test_preds': test_preds,
    }

def rgf_cv_opt(X, y, X_test, folds, n_estimators=100, max_depth=6, max_leaf=63, l1=1, l2=1000, tree_gain_ratio=1):
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'max_leaf': int(max_leaf),
        'l1': float(l1),
        'l2': float(l2),
        'tree_gain_ratio': float(tree_gain_ratio),
    }
    results = rgf_cv(X, y, X_test, folds, params, predict=False, verbose=False)
    mean_cv_score = np.mean(results['cv_scores'])
    return -mean_cv_score


if __name__ == '__main__':
    pass
