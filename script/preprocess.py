import os
import gc
import pandas as pd
from multiprocessing.pool import Pool
from tqdm import tqdm
tqdm.pandas()

n_threads=24

his  = pd.read_csv('../data/historical_transactions.csv', parse_dates=['purchase_date'])
new  = pd.read_csv('../data/new_merchant_transactions.csv', parse_dates=['purchase_date'])
data = pd.concat([his, new], axis=0)
data['merchant_id'] = data['merchant_id'].fillna('__UNKNOWN__MERCH')
data['category_2']  = data['category_2'].fillna('__UNKNOWN__CATEGORY')
data['category_3']  = data['category_3'].fillna('__UNKNOWN__CATEGORY')
del his, new
gc.collect()

os.system('rm -rf ../data/processed_new_transactions_*.csv')
for his_start in range(-5, 0):
    filename = '../data/processed_new_transactions_since_{}.csv'.format(his_start + 1)
    os.system('touch {}'.format(filename))
    with open(filename, 'w') as f:
        f.write('authorized_flag,card_id,city_id,category_1,installments,category_3,merchant_category_id,merchant_id,month_lag,purchase_amount,purchase_date,category_2,state_id,subsector_id\n')

unique_card_ids = data['card_id'].unique().tolist()
card_id_chunks = np.array_split(unique_card_ids, n_threads)

def _process_chunck_customers(pid, card_ids):
    for card_id in tqdm(card_ids, postfix=pid):
        sdf = data.loc[data.card_id == card_id]
        for his_start in range(-13, 0):
            new_start = his_start + 2
            his = sdf.loc[sdf['month_lag'].isin([his_start, his_start + 1])].copy()
            new = sdf.loc[sdf['month_lag'].isin([new_start, new_start + 1])].copy()
            new = new.loc[~new['merchant_id'].isin(his['merchant_id'])]
            new.to_csv('../data/processed_new_transactions_since_{}.csv'.format(his_start + 1), mode='a', header=False, index=False)

p = Pool(n_threads)
for i in range(n_threads):
    p.apply_async(_process_chunck_customers, args=(str(i), card_id_chunks[i]))
p.close()
p.join()
