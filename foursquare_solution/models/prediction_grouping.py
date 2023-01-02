from typing import Dict
import pandas as pd


def to_prediction(df: pd.DataFrame) -> pd.DataFrame:
    preds_all = df.groupby('id').agg({'match_id': set})
    preds_all = preds_all.rename({'match_id': 'all_match_ids'}, axis=1)
    return preds_all


def evaluate_df(preds_all: pd.DataFrame, val_df: pd.DataFrame, id2ids: Dict) -> pd.DataFrame:
    preds = val_df[val_df.prediction == 1].groupby('id').agg({'match_id': set})
    preds['y_true'] = preds.index.map(id2ids.get)
    preds_agg = preds.join(preds_all)
    preds_agg['findable_ids'] = preds_agg.apply(lambda x: x.all_match_ids & x.y_true, axis=1)
    preds_agg['possible_iou'] = preds_agg.apply(lambda x: len(x.findable_ids) / len(x.y_true), axis=1)
    preds_agg['model_iou'] = preds_agg.apply(lambda x: len(x.match_id & x.findable_ids) / len(x.findable_ids), axis=1)
    preds_agg['iou'] = preds_agg.apply(lambda x: len(x.match_id & x.y_true) / len(x.match_id | x.y_true), axis=1)
    preds_agg['maxtruelen'] = preds_agg.y_true.apply(len)
    print(preds_agg[['iou', 'model_iou', 'possible_iou']].mean())
    return preds_agg
