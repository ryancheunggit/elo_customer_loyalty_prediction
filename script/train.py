import os
import argparse
import pickle
import gc; gc.collect()
import numpy as np
import pandas as pd
from feature import *
from model import *
from pprint import pprint
from functools import partial
from datetime import datetime
from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
plt.style.use('fivethirtyeight')


parser = argparse.ArgumentParser(description='parameters for program')
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--recompute', type=int, default=0)
parser.add_argument('--dump_features', type=int, default=0)
parser.add_argument('--n_folds', type=int, default=5)
parser.add_argument('--cv_seed', type=int, default=42)
parser.add_argument('--shift', type=int, default=0)
parser.add_argument('--dataset', type=str, default='final')
parser.add_argument('--model_name', type=str, default='lgb')
parser.add_argument('--task', type=str, default='regression')
parser.add_argument('--param_tuning', type=int, default=0)
args = parser.parse_args()
pprint(vars(args))

def main():
    train, test, target, sub, feature_cols = get_data(shift=args.shift, verbose=args.verbose, recompute=args.recompute, dump_features=args.dump_features)
    if args.verbose: print('-- number of features used: {}'.format(len(feature_cols)))
    if args.task != 'regression': target = target.map(lambda x: 1 if x < -33 else 0)

    # meta_features = get_clf_meta_features(verbose=args.verbose, recompute=args.recompute, seed=42, cut=2)
    # train['__meta__clf_cut_2'] = meta_features['train_meta']
    # test['__meta__clf_cut_2'] = meta_features['test_meta']
    # feature_cols.append('__meta__clf_cut_2')

    folds = get_folds(n_folds=args.n_folds, seed=args.cv_seed, verbose=args.verbose, recompute=False)
    params = get_params(model_name=args.model_name, task=args.task)

    if args.model_name == 'lgb':
        model_cv = lgb_cv
        cv_opt = lgb_cv_opt
        pbounds={
            'max_depth': (4, 12),
            'num_leaves': (63, 255),
            'min_data_in_leaf': (10, 300),
            'feature_fraction': (0.5, 1.0),
            'bagging_fraction': (0.5, 1.0),
            'lambda_l1': (0,10),
            'lambda_l2': (0,10)
        }
    elif args.model_name == 'xgb':
        model_cv = xgb_cv
        cv_opt = xgb_cv_opt
        pbounds={
            'max_depth': (4, 12),
            'min_child_weight': (1, 50),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'gamma': (0, 20),
            'reg_alpha': (0,20),
            'reg_lambda': (0,20)
        }
    elif args.model_name == 'rgf':
        model_cv = rgf_cv
        cv_opt = rgf_cv_opt
        train.fillna(1/1e-10, inplace=True)
        test.fillna(1/1e-10, inplace=True)
        train = train.replace(np.inf, 1/1e-10)
        test = test.replace(np.inf, 1/1e-10)
        train = train.replace(-np.inf, -1/1e-10)
        test = test.replace(-np.inf, -1/1e-10)
        pbounds={
            'n_estimators': (1000, 1000),
            'max_depth': (6, 6),
            'max_leaf': (50, 50),
            'l1': (700, 700),
            'l2': (50, 50),
            'tree_gain_ratio': (0.01, 0.99)
        }

    if args.param_tuning:
        if args.verbose: print('--- tuning parameters for {} rounds'.format(args.param_tuning))
        obj_func = partial(cv_opt, X=train[feature_cols], y=target, X_test=test[feature_cols], folds=folds)
        bayes_opt = BayesianOptimization(f=obj_func, pbounds=pbounds)
        bayes_opt.maximize(init_points=5, n_iter=args.param_tuning, acq='ei', xi=0.0)
        params.update(bayes_opt.max['params'])

    cv_results = model_cv(X=train[feature_cols], y=target, X_test=test[feature_cols], folds=folds, params=params, predict=True, verbose=True)
    if args.task == 'regresion':
        oof_rmse = np.sqrt(np.mean(np.power(target - cv_results['oof_preds'], 2)))
        print('best rmse score at {} : {} - {}'.format(int(np.mean(cv_results['num_iterations'])), np.mean(cv_results['cv_scores']), np.std(cv_results['cv_scores'])))
        print('rmse score based on oof prediction is: {}'.format(oof_rmse))
    if 'fis' in cv_results:
        print(cv_results['fis'][['index', 'mean_feature_importance']].head(20))
        print(cv_results['fis'][['index', 'mean_feature_importance']].tail(10))

    if args.save:
        if args.shift != 0: args.dataset += '_shift_' + str(args.shift)
        date_flag = datetime.now()
        date_flag = '{}-{}-{}-{}'.format(date_flag.month, date_flag.day, date_flag.hour, date_flag.minute)

        trained_model_filepath = '../model/{}_model_{}_data_{}_date_{}.pkl'.format(args.task, args.model_name, args.dataset, date_flag)
        pickle.dump(cv_results['trained_model'], open(trained_model_filepath, 'wb'))
        if args.verbose: print('trained models saved at : {}'.format(trained_model_filepath))

        if 'fis' in cv_results:
            fi_plot_filepath = '../plots/{}_model_{}_data_{}_date_{}.png'.format(args.task, args.model_name, args.dataset, date_flag)
            plt.rcParams['figure.figsize'] = [20, 30]
            plt.tight_layout()
            cv_results['fis'].plot(x='index', y='mean_feature_importance', kind='barh')
            plt.savefig(fi_plot_filepath)
            if args.verbose: print('feature importance plot at : {}'.format(fi_plot_filepath))
            cv_results['fis'].to_csv('../model/{}_model_{}_data_{}_date_{}.csv'.format(args.task, args.model_name, args.dataset, date_flag))

        oof_filepath = '../oof/{}_model_{}_data_{}_date_{}.npy'.format(args.task, args.model_name, args.dataset, date_flag)
        np.save(oof_filepath, cv_results['oof_preds'])
        if args.verbose: print('oof prediction at : {}'.format(oof_filepath))

        sub_filepath = '../submission/{}_model_{}_data_{}_date_{}.csv'.format(args.task, args.model_name, args.dataset, date_flag)
        sub.target = cv_results['test_preds']
        sub.to_csv(sub_filepath, index=False)
        if args.verbose: print('submission file at : {}'.format(sub_filepath))


if __name__ == '__main__':
    main()
