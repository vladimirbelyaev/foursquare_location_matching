import gc
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from foursquare_solution.recall_algos.recall_combinator import recall_all
from foursquare_solution.tools.types_downcasting import cast_booleans, downcast_floats

vec_columns = [
    'name', 'categories', 'address', 'state', 'url', 'country'
]

#boolean_columns = ['found', 'found_name', 'found_country']

def prepare_dataset(data, vecs: np.ndarray, num_neighbors: int, num_neighbors_vec: int, is_train_mode: bool, use_cuda: bool, batch_size: Optional[int]):
    id2index_d = dict(zip(data['id'].values, data.index))

    tfidf_d = {}
    for col in vec_columns:
        tfidf = TfidfVectorizer()
        tv_fit = tfidf.fit_transform(data[col].fillna('nan'))
        tfidf_d[col] = tv_fit

    train_data = recall_all(
        data, 
        [(vecs, '_name', [0.1, 4])], 
        Neighbors=num_neighbors, 
        Neighbors_vec=num_neighbors_vec, 
        use_cuda=use_cuda, 
        batch_size=batch_size
    )
    #cast_booleans(train_data, boolean_columns)
    downcast_floats(train_data)
    train_data.loc[:, 'simple_sim'] = train_data.loc[:, 'simple_sim'].fillna(0).astype(int)


    data = data.set_index('id')


    if is_train_mode:
        ids = train_data['id'].tolist()
        match_ids = train_data['match_id'].tolist()
        poi = data.loc[ids]['point_of_interest'].values
        match_poi = data.loc[match_ids]['point_of_interest'].values
        train_data['label'] = np.array(poi == match_poi, dtype = np.int8)
        print('Num of unique id: %s' % train_data['id'].nunique())
        print('Num of train data: %s' % len(train_data))
        print('Num of positive examples', train_data['label'].sum())
        print('Pos rate: %s' % train_data['label'].mean())
        del ids, match_ids,  poi, match_poi
    gc.collect()
    return train_data, tfidf_d, id2index_d
    

def get_id2poi(input_df: pd.DataFrame) -> dict:
    return dict(zip(input_df['id'], input_df['point_of_interest']))


def get_poi2ids(input_df: pd.DataFrame) -> dict:
    return input_df.groupby('point_of_interest')['id'].apply(set).to_dict()


def get_id2ids(id2poi: dict, poi2ids: dict) -> dict:
    res = dict()
    for _id, poi in id2poi.items():
        res[_id] = poi2ids[poi]
    return res