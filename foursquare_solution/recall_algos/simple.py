import time
from collections import Counter

import pandas as pd
from tqdm.auto import tqdm

recall_columns = ['name', 'address', 'categories', 'address', 'phone']


def recall_simple(df: pd.DataFrame, threshold: float):
    print(time.time(), 'Start recall simple')
    val2id_d = {}
    for col in recall_columns:
        temp_df = df.loc[:, ['id', col]]
        #temp_df[col] = temp_df[col].str.lower()
        val2id = temp_df.groupby(col)['id'].apply(set).to_dict()
        val2id_d[col] = val2id
        del val2id
    cus_ids = []
    match_ids = []
    values = []
    for vals in tqdm(df[recall_columns + ['id']].fillna('null').values):
        cus_id = vals[-1]
        match_id = []

        rec_match_count = []
        for i in range(len(recall_columns)):
            col = recall_columns[i]

            if vals[i] != 'null':
                rec_match_count += list(val2id_d[col][vals[i].lower()])
        rec_match_count = dict(Counter(rec_match_count))

        for k, v in rec_match_count.items():
            if v > threshold:
                match_id.append(k)
                values.append(v)

        cus_ids += [cus_id] * len(match_id)
        match_ids += match_id

    train_df = pd.DataFrame()
    train_df['id'] = cus_ids
    train_df['match_id'] = match_ids
    train_df['simple_sim'] = values
    train_df = train_df.drop_duplicates()
    del cus_ids, match_ids

    num_data = len(train_df)
    num_data_per_id = num_data / train_df['id'].nunique()
    print('Num of data: %s' % num_data)
    print('Num of data per id: %s' % num_data_per_id)

    return train_df