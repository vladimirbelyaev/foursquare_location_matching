from itertools import combinations
from typing import Dict
import pandas as pd
import numpy as np
from foursquare_solution.features.distance import haversine_np


THRESHOLD_OWN_VALUES = 1000
DEFAULT_KEY = '<DEFAULT>'


def avg_haversine_country(data: pd.DataFrame) -> dict:
    poi_all_pairs = data.groupby(['point_of_interest'])['id'].apply(lambda x: list(combinations(x, 2))).explode().dropna().values
    poi_all_pairs = np.vstack(poi_all_pairs)
    data = data.set_index('id')
    df_pairs = pd.DataFrame(poi_all_pairs, columns=['id', 'match_id'])
    df_pairs['country'] = data.loc[df_pairs.id, 'country'].values
    df_pairs['haversine_dist'] = haversine_np(data.loc[df_pairs.id, 'longitude'].values, data.loc[df_pairs.id, 'latitude'].values, data.loc[df_pairs.match_id, 'longitude'].values, data.loc[df_pairs.match_id, 'latitude'].values)
    agg_overall = df_pairs.haversine_dist.median()
    agg_country = df_pairs.groupby('country').agg({'haversine_dist': 'median', 'id': 'count'})
    agg_country = agg_country[agg_country.id >= THRESHOLD_OWN_VALUES]
    dct = {DEFAULT_KEY: agg_overall}
    dct.update(agg_country.haversine_dist.to_dict())
    return dct


def avg_neighbors_country(data: pd.DataFrame) -> dict:
    agg_overall = data.groupby('point_of_interest')['id'].count().mean()
    agg_country = data.groupby('point_of_interest').agg({'country': 'first',  'id': 'count'}).groupby('country').agg({'id':['mean', 'count']})
    agg_country = agg_country[agg_country[('id', 'count')] >= THRESHOLD_OWN_VALUES]
    dct = {DEFAULT_KEY: agg_overall}
    dct.update(agg_country[('id', 'mean')].to_dict())
    return dct


def add_country_features(train_data: pd.DataFrame, haversine_dct: Dict, neighbors_dct: Dict) -> None:
    train_data.loc[:, 'country_avg_haversine'] = train_data.loc[:, 'country'].apply(lambda x: haversine_dct.get(x, haversine_dct[DEFAULT_KEY]))
    train_data.loc[:, 'country_avg_haversine_diff'] = train_data.haversine_dist - train_data.country_avg_haversine
    train_data.loc[:, 'country_avg_haversine_div'] = train_data.haversine_dist / train_data.country_avg_haversine
    train_data.loc[:, 'country_avg_neighbors'] = train_data.loc[:, 'country'].apply(lambda x: neighbors_dct.get(x, neighbors_dct[DEFAULT_KEY]))
