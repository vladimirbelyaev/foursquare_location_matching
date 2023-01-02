from typing import Dict, Optional
import numpy as np
import pandas as pd
import h3

from foursquare_solution.tools.types_downcasting import downcast_floats


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    return c


def l2(lon1, lat1, lon2, lat2):
    return np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)


def recalculate_distances(
    df: pd.DataFrame, data: pd.DataFrame, hexagon2counts: Optional[Dict[str, float]] = None
    ):
    lon1, lat1 = data.loc[df['id']]['longitude'].values, data.loc[df['id']]['latitude'].values
    lon2, lat2 = data.loc[df['match_id']]['longitude'].values, data.loc[df['match_id']]['latitude'].values
    df.loc[:, 'longitude'] = lon1
    df.loc[:, 'latitude'] = lat1
    df.loc[:, 'h3_7'] = [h3.geo_to_h3(lat, lng, 7) for lat, lng in zip(df['latitude'], df['longitude'])]
    if hexagon2counts is not None:
        df.loc[:, 'h3_7_density'] = df.h3_7.map(hexagon2counts)
    df.loc[:, 'haversine_dist'] = haversine_np(lon1, lat1, lon2, lat2)
    df.loc[:, 'euclidian_dist'] = l2(lon1, lat1, lon2, lat2)
    downcast_floats(df)
    return df
