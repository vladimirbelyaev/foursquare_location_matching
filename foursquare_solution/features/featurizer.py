import gc
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from foursquare_solution.features.agg_features import add_country_features

from foursquare_solution.features.distance import recalculate_distances
from foursquare_solution.features.text import calculate_feature, feat_columns
from foursquare_solution.features.text_vec import recalculate_text_distances
from foursquare_solution.tools.types_downcasting import downcast_floats


def add_features_vectorized(
    df, data, tfidf_d, id2index_d, vecs, avg_hvs, avg_nbs, hexagon2counts
):
    # df['country'] = data.loc[df['id']].country.values.fillna('')
    df.loc[:, 'country'] = data.loc[df['id'], 'country'].fillna('').values
    df.loc[:, 'country'] = df.loc[:, 'country'].astype("category")
    print(time.time(), 'recalculating distances')
    df = recalculate_distances(df, data, hexagon2counts)
    add_country_features(df, avg_hvs, avg_nbs)
    df = recalculate_text_distances(df, data, vecs)

    gc.collect()
    print('featurer on')
    for col in tqdm(feat_columns):
        df = calculate_feature(df, data, tfidf_d, id2index_d, col)
        downcast_floats(df)
        gc.collect()
    return df


def create_ds_with_feats(
    data: pd.DataFrame,
    train_data: pd.DataFrame,
    vecs: np.ndarray,
    tfidf_d,
    id2index_d,
    avg_hvs,
    avg_nbs,
    hexagon2counts,
    mode,
    num_splits: int,
    num_splits_to_save: int
):
    count = 0
    start_row = 0
    # print(data.index)
    data = data.set_index('id')
    unique_id = train_data['id'].unique().tolist()
    num_split_id = len(unique_id) // num_splits
    for k in range(num_splits_to_save):
        print('Current split: %s' % k)
        end_row = start_row + num_split_id
        if k < num_splits:
            cur_id = unique_id[start_row:end_row]
            cur_data = train_data[train_data['id'].isin(cur_id)]
        else:
            cur_id = unique_id[start_row:]
            cur_data = train_data[train_data['id'].isin(cur_id)]

        cur_data = add_features_vectorized(cur_data, data, tfidf_d, id2index_d, vecs, avg_hvs, avg_nbs, hexagon2counts)

        print(cur_data.shape)
        print(cur_data.sample(1))
        start_row = end_row
        count += len(cur_data)
        yield cur_data
        del cur_data
        gc.collect()

    print(count)