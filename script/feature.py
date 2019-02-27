import os
import pickle
import gc; gc.enable()
import numpy as np
import pandas as pd
from tqdm import tqdm; tqdm.pandas()
from functools import partial
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from model import *

EPSILON = 1e-10

def get_folds(n_folds=5, seed=42, verbose=0, recompute=0):
    folds_path = '../data/cv_folds_seed_{}.pkl'.format(seed)
    if not recompute and os.path.exists(folds_path):
        if verbose: print('load cv splits from dump {}'.format(folds_path))
        folds = pickle.load(open(folds_path, 'rb'))
    else:
        if verbose: print('compute cv splits and save to {}'.format(folds_path))
        train = pd.read_csv('../data/train.csv', parse_dates=['first_active_month'])
        target = train['target']
        loyalty_stratification = target.map(lambda x: 1 if x < -33 else 0)
        folds = list(
            StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=seed
            ).split(train, loyalty_stratification)
        )
        pickle.dump(folds, open(folds_path, 'wb'))
    return folds


def get_data(shift=0, verbose=0, recompute=0, dump_features=0):
    train = pd.read_csv('../data/train.csv', parse_dates=['first_active_month'])
    test = pd.read_csv('../data/test.csv', parse_dates=['first_active_month'])
    sub = pd.read_csv('../data/sample_submission.csv')
    num_train = train.shape[0]
    target = train['target']
    outlier = target.map(lambda x: 1 if x < -33 else 0)
    train.drop('target', axis=1, inplace=True)
    test['first_active_month'].fillna(pd.Timestamp('2017-03-09'), inplace=True)

    data = pd.concat([train, test], axis=0)

    reference_date = data['first_active_month'].min()
    data['__days__customer_first_aquired'] = (data['first_active_month'] - reference_date).dt.days.astype(int)

    date_features = get_date_features(shift=shift, verbose=verbose, recompute=recompute, dump_features=dump_features)
    data = data.merge(date_features, on='card_id', how='left')

    data['__date__observation'] = (data['cutoff_date'] - reference_date).dt.days
    data['__date__recency'] = data['__date__observation'] - data['__days__customer_first_aquired']

    for feature in ['feature_1', 'feature_2', 'feature_3']:
        outlier_priori = outlier.groupby(train[feature]).mean()
        # --- try to ingest a bit priori
        data['__priori__outlier_+_' + feature] = data['feature_1'].map(outlier_priori)
        # --- recency interact with feature ?? would make sense if feature is somekind of credit ranking of card
        data['__prod__date__recency_+_' + feature] = data['__date__recency'] * data[feature]
        data['__ratio__date__recency_+_' + feature] = data[feature] / data['__date__recency']

    data['__encode__three_features'] = data[['feature_1', 'feature_2', 'feature_3']].\
        progress_apply(lambda x: '_'.join([str(e) for e in x]), axis=1)
    three_features_count_encoding = data['__encode__three_features'].value_counts().to_dict()
    data['__encode__three_features'] = data['__encode__three_features'].map(three_features_count_encoding)
    data['__prod__three_features'] = data[['feature_1', 'feature_2', 'feature_3']].product(axis=1)

    transaction_features = get_transaction_features(shift=shift, verbose=verbose, recompute=recompute, dump_features=dump_features)
    data = data.merge(right=transaction_features, on='card_id', how='left')

    data['__num__purchases_ratio'] = data['new__num__purchases'] / (data['his__num__purchases'] + EPSILON)
    data['__ratio__total_purchase_amount'] = data['new__total__purchase_amount'] / (data['his__total__purchase_amount'] + EPSILON)
    data['__mean__purchase_amount_diff'] = data['new__mean__purchase_amount'] - data['his__mean__purchase_amount']
    data['__ratio__mean__dates_diff'] = data['new__mean__purchase_days_diff'] / (data['his__mean__purchase_days_diff'] + EPSILON)
    # data.fillna(value=-99999999, inplace=True)

    # new = pd.read_csv('../data/new_merchant_transactions.csv')
    # data['__bool__has_new'] = data['card_id'].isin(new.card_id).astype(int)

    train = data.iloc[:num_train, :]
    test = data.iloc[num_train:, :]

    feature_cols = ['feature_1', 'feature_2', 'feature_3'] + [colname
        for colname in train.columns
        if '__' in colname
    ]

    return train, test, target, sub, feature_cols


def get_transaction_features(shift=0, verbose=0, recompute=0, dump_features=0):
    feature_file_path = '../data/features/transaction_features_{}.pkl'.format(shift)
    if not recompute and os.path.exists(feature_file_path):
        if verbose: print('--- read transactions features from dump')
        features = pd.read_pickle(feature_file_path)
    else:
        if shift == 0:
            his = pd.read_csv('../data/historical_transactions.csv', parse_dates=['purchase_date'])
            new = pd.read_csv('../data/new_merchant_transactions.csv', parse_dates=['purchase_date'])
        else:
            his = pd.read_csv('../data/historical_transactions.csv', parse_dates=['purchase_date'])
            his = his.loc[his['month_lag'] <= shift]
            new = pd.read_csv('../data/processed_new_transactions_since_{}.csv'.format(shift), parse_dates=['purchase_date'])
            new = new.loc[new['authorized_flag'] == 'Y']
            his['month_lag'] -= shift
            new['month_lag'] -= shift
        his['dataset'] = 'his'
        new['dataset'] = 'new'
        data = pd.concat([his, new], axis=0)
        data['merchant_id'] = data['merchant_id'].fillna('UNKNOWN_MERCHANT').astype(str)
        data['category_2']  = data['category_2'].fillna('__UNKNOWN__CATEGORY').astype(str)
        data['category_3']  = data['category_3'].fillna('__UNKNOWN__CATEGORY').astype(str)

        # - recover the purchase amount to the dollar amount
        data['purchase_amount'] = np.round(data['purchase_amount'] / 0.00150265118 + 497.06, 2)

        # - feature generation
        # -- aggregate on purchase amount
        if verbose: print('--- generate overall purchase amount features: {}'.format(datetime.now()))
        card_overall_agg_colnames = [
            '__num__purchases',
            '__total__purchase_amount',
            '__mean__purchase_amount',
            '__min__purchase_amount',
            '__max__purchase_amount',
            '__std__purchase_amount',
            '__skew__purchase_amount'
        ]
        features = data.groupby(['card_id', 'dataset'])['purchase_amount'].\
            agg(['size', 'sum', 'mean', 'min', 'max', 'std', 'skew']).\
            reset_index().\
            pivot(index='card_id', columns='dataset').\
            reset_index()
        features.columns = ['card_id'] + [
            '{}{}'.format(dataset, colname)
            for colname in card_overall_agg_colnames
            for dataset in ['his', 'new']
        ]

        # -- aggregate over month_lag
        if verbose: print('--- generate by month_lag purchase amount features: {}'.format(datetime.now()))
        def _count_num_zero(ts):
            return len(ts.nonzero()[0])

        low, high = data['month_lag'].min(), data['month_lag'].max() + 1
        card_by_month_agg = data.groupby(['card_id', 'month_lag'])['purchase_amount'].\
            agg(['size', 'sum', 'mean']).\
            reset_index().\
            pivot(index='card_id', columns='month_lag').\
            reset_index()
        card_by_month_agg.columns = ['card_id'] + ['__{}__purchases_during_month_lag_{}'.format(agg, month_lag) for month_lag in range(low, high) for agg in ['num', 'sum', 'mean']]
        card_by_month_agg.fillna(0, inplace=True)
        features = features.merge(right=card_by_month_agg, on='card_id', how='left')

        # --- recurrent purchase

        def _get_recurrent_purchase(sdf):
            vc = sdf['purchase_amount'].value_counts().reset_index().sort_values('purchase_amount').iloc[-5:]
            ret = {
                '__rec_cnt_1': 0,
                '__rec_cnt_2': 0,
                '__rec_cnt_3': 0,
                # '__rec_cnt_4': 0,
                # '__rec_cnt_5': 0,
                '__rec_amt_1': 0,
                '__rec_amt_2': 0,
                '__rec_amt_3': 0,
                # '__rec_amt_4': 0,
                # '__rec_amt_5': 0,
            }
            if vc.shape[0] >= 1:
                ret['__rec_cnt_1'] = vc.iloc[-1, 1]
                ret['__rec_amt_1'] = vc.iloc[-1, 0]
            if vc.shape[0] >= 2:
                ret['__rec_cnt_2'] = vc.iloc[-2, 1]
                ret['__rec_amt_2'] = vc.iloc[-2, 0]
            if vc.shape[0] >= 3:
                ret['__rec_cnt_3'] = vc.iloc[-2, 1]
                ret['__rec_amt_3'] = vc.iloc[-2, 0]
            # if vc.shape >= 4:
            #     ret['__rec_cnt_4'] = vc.iloc[-2, 1]
            #     ret['__rec_amt_4'] = vc.iloc[-2, 0]
            # if vc.shape >= 5:
            #     ret['__rec_cnt_5'] = vc.iloc[-2, 1]
            #     ret['__rec_amt_5'] = vc.iloc[-2, 0]
            return pd.Series(ret)

        for his_length in [1,2,3,5,8]:
            hdata = his.loc[his.month_lag > -his_length]
            rec_feature = hdata.groupby(['card_id']).progress_apply(_get_recurrent_purchase)
            rec_feature.columns = [col + '_his_length_{}'.format(his_length) for col in rec_feature.columns]
            rec_feature = rec_feature.reset_index()
            features = features.merge(right=rec_feature, how='left')

        for lag in [3, 6, 9]:
            for fshift in [0, 1, 2]:
                try:
                    suffix = '_lag_{}_shift_{}'.format(lag, fshift)
                    cols = ['__sum__purchases_during_month_lag_{}'.format(- l - fshift) for l in range(lag)]
                    features['__sum__purchases' + suffix] = features[cols].sum(axis=1)
                    features['__num__active_months'  + suffix] = features[cols].progress_apply(_count_num_zero, axis=1)
                    features['__avg__active_months_purchases' + suffix] = features['__sum__purchases' + suffix] / (features['__num__active_months' + suffix] + EPSILON)
                    features['__ratio__active_months_purchases' + suffix] = features['__sum__purchases_during_month_lag_{}'.format(0 - fshift)] / (features['__avg__active_months_purchases' + suffix] + EPSILON)
                # features['__ratio__last_2_active_months_purchases' + suffix] = (features['__sum__purchases_during_month_lag_{}'.format(0 - fshift)] + features['__sum__purchases_during_month_lag_{}'.format(0 - fshift - 1)])/ (features['__avg__active_months_purchases' + suffix] + EPSILON)
                except:
                    pass

        # -- number of unauthorized transactions; note there is no unauthorized purchases in new transactions!
        if verbose: print('--- generate unauthorized purchase features: {}'.format(datetime.now()))
        num_unauthorized_transactions = data.loc[data['authorized_flag'] == 'N'].\
            groupby(['card_id', 'month_lag']).\
            size().\
            reset_index().\
            pivot(index='card_id', columns='month_lag').\
            reset_index().\
            fillna(0)
        num_unauthorized_transactions.columns = ['card_id'] + ['__num__unauthorized_transactions_lag_{}'.format(i) for i in range(low, 1)]
        num_unauthorized_transactions['__num__total_unauthorized_transactions'] = num_unauthorized_transactions.iloc[:, 1:].sum(axis=1)
        features = features.merge(right=num_unauthorized_transactions, on='card_id', how='left')
        for month_lag in range(low, 1):
            features['__ratio__unauthorized_transactions_lag_{}'.format(month_lag)] = \
                features['__num__unauthorized_transactions_lag_{}'.format(month_lag)] / \
                (features['__num__purchases_during_month_lag_{}'.format(month_lag)] + EPSILON)
        features['__ratio__total_unauthorized_transactions'] = features['__num__total_unauthorized_transactions'] / (features['his__num__purchases'] + EPSILON)

        # -- number -> ration of purchases in different category 1
        if verbose: print('--- generate category faceted features: {}'.format(datetime.now()))
        def _cnt_agg_by_cat(category):
            category_gpb_agg = data.groupby(['card_id', 'dataset', category]).size().unstack().reset_index().pivot(index='card_id', columns='dataset').reset_index().fillna(0)
            category_gpb_agg.columns = category_gpb_agg.columns.map(lambda x: '_'.join(str(e) for e in x)).str.strip('_')
            category_gpb_agg.columns = ['card_id'] + ['__ratio_{}_{}'.format(category, val) for val in category_gpb_agg.columns[1:]]
            category_gpb_agg['his__nunique__{}'.format(category)] = category_gpb_agg[[col for col in category_gpb_agg.columns if 'ratio' in col and 'his' in col]].progress_apply(_count_num_zero, axis=1)
            category_gpb_agg['new__nunique__{}'.format(category)] = category_gpb_agg[[col for col in category_gpb_agg.columns if 'ratio' in col and 'his' in col]].progress_apply(_count_num_zero, axis=1)
            category_gpb_agg = features[['card_id', 'his__num__purchases', 'new__num__purchases']].merge(right=category_gpb_agg, on='card_id', how='left')
            category_gpb_agg.fillna(0, inplace=True)
            for col in category_gpb_agg.columns:
                if 'ratio' in col:
                    if 'his' in col:
                        category_gpb_agg[col] /= (category_gpb_agg['his__num__purchases'] + EPSILON)
                    if 'new' in col:
                        category_gpb_agg[col] /= (category_gpb_agg['new__num__purchases'] + EPSILON)
            category_gpb_agg.drop('his__num__purchases', axis=1, inplace=True)
            category_gpb_agg.drop('new__num__purchases', axis=1, inplace=True)
            return category_gpb_agg

        category_1_agg = _cnt_agg_by_cat('category_1')
        category_2_agg = _cnt_agg_by_cat('category_2')
        category_3_agg = _cnt_agg_by_cat('category_3')
        features = features.merge(right=category_1_agg, on='card_id', how='left')
        features = features.merge(right=category_2_agg, on='card_id', how='left')
        features = features.merge(right=category_3_agg, on='card_id', how='left')

        def _sum_amt_agg_by_cat(category):
            category_gpb_agg = data.groupby(['card_id', 'dataset', category]).agg({'purchase_amount': 'sum'}).unstack().reset_index().pivot(index='card_id', columns='dataset').reset_index().fillna(0)
            category_gpb_agg.columns = category_gpb_agg.columns.map(lambda x: '_'.join(str(e) for e in x)).str.strip('_')
            category_gpb_agg.columns = ['card_id'] + ['__sum_{}_{}'.format(category, val) for val in category_gpb_agg.columns[1:]]
            category_gpb_agg = features[['card_id', 'his__total__purchase_amount', 'new__total__purchase_amount']].merge(right=category_gpb_agg, on='card_id', how='left')
            category_gpb_agg.fillna(0, inplace=True)
            for col in category_gpb_agg.columns:
                if 'sum' in col:
                    if 'his' in col:
                        category_gpb_agg[col] /= (category_gpb_agg['his__total__purchase_amount'] + EPSILON)
                    if 'new' in col:
                        category_gpb_agg[col] /= (category_gpb_agg['new__total__purchase_amount'] + EPSILON)
            category_gpb_agg.drop('his__total__purchase_amount', axis=1, inplace=True)
            category_gpb_agg.drop('new__total__purchase_amount', axis=1, inplace=True)
            return category_gpb_agg

        category_1_agg = _sum_amt_agg_by_cat('category_1')
        category_2_agg = _sum_amt_agg_by_cat('category_2')
        category_3_agg = _sum_amt_agg_by_cat('category_3')
        features = features.merge(right=category_1_agg, on='card_id', how='left')
        features = features.merge(right=category_2_agg, on='card_id', how='left')
        features = features.merge(right=category_3_agg, on='card_id', how='left')

        # - how often and how much customer goes for installment purchasing
        # -- how many installment transactions
        if verbose: print('--- generate installments related features: {}'.format(datetime.now()))
        installment_transactions = data.groupby(['card_id'])['installments'].agg(lambda x: len([e for e in x if e >=1 and e != 999])).reset_index()
        installment_transactions.columns = ['card_id', '__num__installment_transactions']
        # -- total amount with non-installment / installment purchases
        installment_transactions_amount_ratio = data[['card_id', 'installments', 'purchase_amount']].copy()
        installment_transactions_amount_ratio['installments'] = installment_transactions_amount_ratio['installments'].map(lambda x: 1 if x >=1 and x != 999 else 0)
        installment_transactions_amount_ratio = installment_transactions_amount_ratio.groupby(['card_id', 'installments'])['purchase_amount'].sum().reset_index().pivot(index='card_id', columns='installments', values='purchase_amount').fillna(0).reset_index()
        installment_transactions_amount_ratio.columns = ['card_id', '__sum__non_installment_purchase_amount', '__sum__installment_purchase_amount']
        installment_transactions_amount_ratio['__ratio__installment_purchase_amount'] = installment_transactions_amount_ratio['__sum__installment_purchase_amount'] / (installment_transactions_amount_ratio['__sum__non_installment_purchase_amount'] + EPSILON)
        installment_transactions = installment_transactions.merge(right=installment_transactions_amount_ratio, on='card_id', how='left')
        installment_transactions['__mean__installment_purchase_amount'] = installment_transactions['__sum__installment_purchase_amount'] / (installment_transactions['__num__installment_transactions'] + EPSILON)
        features = features.merge(right=installment_transactions, on='card_id', how='left')

        # -- most frequent merchant / merchant category / subsector id -> what the customer buy most frequently
        if verbose: print('--- generate merchant cat/id/ sector related features: {}'.format(datetime.now()))
        def _get_mode(ts):
            return ts.mode()[0]
        customer_favorate = data.groupby(['card_id']).agg({
            'merchant_category_id': ['nunique', _get_mode],
            'merchant_id': ['nunique', _get_mode],
            'subsector_id': ['nunique', _get_mode],
        }).reset_index()
        customer_favorate.columns = [
            'card_id',
            '__nunique__merchant_cat', '__mode__merchant_cat',
            '__nunique__merchant_id', '__mode__merchant_id',
            '__nunique__subsector_id', '__mode__subsector_id',
        ]
        customer_favorate['__mode__merchant_id'] = customer_favorate['__mode__merchant_id'].map(lambda x: x if x != -1 else 'UNKNOWN_MERCHANT')
        merchant_id_mapping = customer_favorate['__mode__merchant_id'].value_counts().to_dict()
        customer_favorate['__mode__merchant_id'] = customer_favorate['__mode__merchant_id'].map(merchant_id_mapping)

        features = features.merge(right=customer_favorate, on='card_id', how='left')

        # - some ratio
        features['__freq__installment_purchases'] = features['__num__installment_transactions'] / (features['his__num__purchases'] + EPSILON)

    if dump_features:
        features.to_pickle(feature_file_path)
    return features


def get_date_features(shift=0, verbose=0, recompute=0, dump_features=0):
    def __unique_days_diff_agg(ts):
        ts = pd.Series(ts.unique()).diff()
        return pd.Series({'mean': ts.mean(), 'min': ts.min(), 'max': ts.max(), 'std': ts.std(), 'skew': ts.skew()})

    diff_agg_names = ['__mean__', '__min__', '__max__', '__std__', '__skew__']
    feature_file_path = '../data/features/date_features_{}.pkl'.format(shift)

    if not recompute and os.path.exists(feature_file_path):

        if verbose: print('--- load date feature from dump')
        features = pd.read_pickle(feature_file_path)
    else:
        if verbose: print('--- compute date feature from scratch')
        if shift == 0:
            his = pd.read_csv('../data/historical_transactions.csv', parse_dates=['purchase_date'])
            new = pd.read_csv('../data/new_merchant_transactions.csv', parse_dates=['purchase_date'])
        else:
            his = pd.read_csv('../data/historical_transactions.csv', parse_dates=['purchase_date'])
            his = his.loc[his['month_lag'] <= shift]
            new = pd.read_csv('../data/processed_new_transactions_since_{}.csv'.format(shift), parse_dates=['purchase_date'])
            new = new.loc[new['authorized_flag'] == 'Y']
            his['month_lag'] -= shift
            new['month_lag'] -= shift
        reference_date = his['purchase_date'].min()
        his['purchase_ref_days'] = (his['purchase_date'] - reference_date).dt.days
        new['purchase_ref_days'] = (new['purchase_date'] - reference_date).dt.days

        if verbose: print('--- recovering actual cutoff date / get min max date --- {}'.format(datetime.now()))
        card_his = his.groupby('card_id').agg({'month_lag': 'max', 'purchase_date': ['max', 'min']}).reset_index()
        card_his.columns = ['card_id', 'max_month_lag', 'his_latest_purchase_date', 'his_earliest_purchase_date']
        card_his['cutoff_date'] = card_his.progress_apply(lambda x: x['his_latest_purchase_date']  - pd.DateOffset(months=x['max_month_lag']), axis=1)
        card_his['cutoff_date'] = card_his['cutoff_date'].dt.to_period('M').dt.to_timestamp() + pd.DateOffset(months=1)
        card_new = new.groupby('card_id').agg({'purchase_date': ['max', 'min']}).reset_index()
        card_new.columns = ['card_id', 'new_latest_purchase_date', 'new_earliest_purchase_date']

        if verbose: print('--- getting days diff features on his --- {}'.format(datetime.now()))
        his.sort_values(by=['card_id', 'purchase_date'], inplace=True)
        his = his.loc[his['month_lag'] >= -8]

        his['purchase_days_diff'] = his.groupby('card_id')['purchase_date'].diff().dt.days
        his_purchase_days_diff_agg = his.groupby('card_id')['purchase_days_diff'].agg(['mean', 'min', 'max', 'std', 'skew']).reset_index()
        his_purchase_days_diff_agg.columns = ['card_id'] + ['his{}purchase_days_diff'.format(agg_name) for agg_name in diff_agg_names]

        his['purchase_hours_diff'] = his.groupby('card_id')['purchase_date'].diff().dt.seconds // 3600
        his_purchase_hours_diff_agg = his.groupby('card_id')['purchase_hours_diff'].agg(['mean', 'min', 'max', 'std', 'skew']).reset_index()
        his_purchase_hours_diff_agg.columns = ['card_id'] + ['his{}purchase_hours_diff'.format(agg_name) for agg_name in diff_agg_names]

        his_purchase_ref_days_diff_agg = his.groupby('card_id')['purchase_ref_days'].progress_apply(__unique_days_diff_agg).reset_index().pivot(index='card_id', columns='level_1', values='purchase_ref_days').reset_index()
        his_purchase_ref_days_diff_agg.columns = ['card_id'] + ['his{}purchase_ref_days_diff'.format(agg_name) for agg_name in diff_agg_names]

        card_his = card_his.merge(right=his_purchase_days_diff_agg, on='card_id', how='left')
        card_his = card_his.merge(right=his_purchase_hours_diff_agg, on='card_id', how='left')
        card_his = card_his.merge(right=his_purchase_ref_days_diff_agg, on='card_id', how='left')

        if verbose: print('--- getting days diff features on new --- {}'.format(datetime.now()))
        new.sort_values(by=['card_id', 'purchase_date'], inplace=True)

        new['purchase_days_diff'] = new.groupby('card_id')['purchase_date'].diff().dt.days
        new_purchase_days_diff_agg = new.groupby('card_id')['purchase_days_diff'].agg(['mean', 'min', 'max', 'std', 'skew']).reset_index()
        new_purchase_days_diff_agg.columns = ['card_id'] + ['new{}purchase_days_diff'.format(agg_name) for agg_name in diff_agg_names]

        new['purchase_hours_diff'] = new.groupby('card_id')['purchase_date'].diff().dt.seconds // 3600
        new_purchase_hours_diff_agg = new.groupby('card_id')['purchase_hours_diff'].agg(['mean', 'min', 'max', 'std', 'skew']).reset_index()
        new_purchase_hours_diff_agg.columns = ['card_id'] + ['new{}purchase_hours_diff'.format(agg_name) for agg_name in diff_agg_names]

        new_purchase_ref_days_diff_agg = new.groupby('card_id')['purchase_ref_days'].progress_apply(__unique_days_diff_agg).reset_index().pivot(index='card_id', columns='level_1', values='purchase_ref_days').reset_index()
        new_purchase_ref_days_diff_agg.columns = ['card_id'] + ['new{}purchase_ref_days_diff'.format(agg_name) for agg_name in diff_agg_names]

        card_new = card_new.merge(right=new_purchase_days_diff_agg, on='card_id', how='left')
        card_new = card_new.merge(right=new_purchase_hours_diff_agg, on='card_id', how='left')
        card_new = card_new.merge(right=new_purchase_ref_days_diff_agg, on='card_id', how='left')

        if verbose: print('--- putting things all together --- {}'.format(datetime.now()))
        features = card_his.merge(right=card_new, on='card_id', how='left')

        features['__diff__his_new_date_range'] = (features['new_latest_purchase_date'] - features['his_earliest_purchase_date']).dt.days
        features['__diff__his_new_date_gap'] = (features['new_earliest_purchase_date'] - features['his_latest_purchase_date']).dt.days
        features['__diff__his_obs_date_range'] = (features['cutoff_date'] - features['his_earliest_purchase_date']).dt.days
        features['__diff__his_obs_date_gap'] = (features['cutoff_date'] - features['his_latest_purchase_date']).dt.days
        features['__diff__new_obs_date_range'] = (features['new_latest_purchase_date'] - features['cutoff_date']).dt.days
        features['__diff__new_obs_date_gap'] = (features['new_earliest_purchase_date'] - features['cutoff_date']).dt.days

    if dump_features:
        features.to_pickle(feature_file_path)
    return features


def get_clf_meta_features(verbose=0, recompute=0, dump_features=0, seed=1412, cut=2):
    feature_filepath = '../data/features/clf_meta_seed_{}_cut_{}.pkl'.format(seed, cut)
    if not recompute and os.path.exists(feature_filepath):
        if verbose: print('load meta feature from dump {}'.format(feature_filepath))
        features = pickle.load(open(feature_filepath, 'rb'))
    else:
        if verbose: print('compute meta feature and dump to {}'.format(feature_filepath))
        train, test, target, sub, feature_cols = get_data(verbose=0, recompute=0)
        folds = get_folds(n_folds=5, seed=seed, verbose=verbose, recompute=recompute)
        if cut == 2:
            target = target.map(lambda x: 1 if x < -32 else 0)
        else:
            target = pd.cut(target, bins=[-36, -18, -0.5, 0.5, 18], labels=[0,1,2,3]).astype(int)

        cv_results = rfc_cv(
            X=train[feature_cols],
            y=target,
            X_test=test[feature_cols],
            folds=folds,
            params=get_params('rfc'),
            predict=True,
            verbose=verbose
        )
        features = {
            'train_meta': cv_results['oof_preds'],
            'test_meta': cv_results['test_preds']
        }
    if dump_features:
        pickle.dump(features, open(feature_filepath, 'wb'))
    return features


if __name__ == '__main__':
    pass
