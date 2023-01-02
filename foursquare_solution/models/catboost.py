import pandas as pd

import catboost as cb
from foursquare_solution.models.features import train_cols, cat_features


def make_pool(df: pd.DataFrame, has_label: bool) -> cb.Pool:
    return cb.Pool(
    df[train_cols],
    label=df.label.values if has_label else None,
    feature_names=train_cols,
    cat_features=cat_features
)
