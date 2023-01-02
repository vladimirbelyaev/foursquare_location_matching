import time

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree, KDTree
from tqdm.auto import tqdm

from foursquare_solution.recall_algos.neighbors import (get_basic_neighbors,
                                                        get_kdtree_neighbors)


def recall_knn_country(df, Neighbors=10, enhanced=True, convert_to_radians=True):
    print(time.time(), 'Start knn grouped by country')
    vec_fields = ['latitude', 'longitude']
    train_df_country = []
    for country, country_df in tqdm(df.groupby('country')):
        country_df = country_df.reset_index(drop = True)

        neighbors = min(len(country_df), Neighbors)
        vecs = country_df[vec_fields]
        if convert_to_radians:
            vecs = np.deg2rad(vecs)
        if enhanced:
            cur_df = get_kdtree_neighbors(
                country_df, vecs, tree_class=BallTree, base_neighbors=neighbors,
                metric='haversine', tag='_country', max_threshold=0.01
            )
        else:
            cur_df = get_basic_neighbors(
                country_df, vecs, neighbors=neighbors, metric='haversine', tag='_country'
                )
        train_df_country.append(cur_df)
    train_df_country = pd.concat(train_df_country)
    return train_df_country


def recall_knn_basic(df, Neighbors=10, enhanced=True, convert_to_radians=True):
    print(time.time(), 'Start basic knn')
    vec_fields = ['latitude', 'longitude']
    vecs = df[vec_fields]
    if convert_to_radians:
        vecs = np.deg2rad(vecs)
    if enhanced:
        train_df = get_kdtree_neighbors(
            df, vecs, tree_class=KDTree, base_neighbors=Neighbors, max_threshold=0.01
        )
    else:
        train_df = get_basic_neighbors(df, vecs, neighbors=Neighbors)
    return train_df