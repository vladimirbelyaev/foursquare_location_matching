
import gc
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def analysis(df: pd.DataFrame):
    print('First idxs:', df.id.head(5).tolist())
    print('Num of data: %s' % len(df))
    print('Num of unique id: %s' % df['id'].nunique())
    print('Num of unique poi: %s' % df['point_of_interest'].nunique())

    poi_grouped = df.groupby('point_of_interest')['id'].count().reset_index()
    print('Mean num of unique poi: %s' % poi_grouped['id'].mean())


def get_score(input_df: pd.DataFrame, id2poi: dict, poi2ids: dict):
    scores = []
    for id_str, matches in zip(input_df['id'].to_numpy(), input_df['matches'].to_numpy()):
        targets = poi2ids[id2poi[id_str]]
        preds = set(matches.split())
        score = len((targets & preds)) / len((targets | preds))
        scores.append(score)
    scores = np.array(scores)
    return scores.mean()


def get_score_and_nonideal(input_df: pd.DataFrame, id2poi: dict, poi2ids: dict):
    nonideal = []
    scores = []
    for id_str, matches in zip(input_df['id'].to_numpy(), input_df['matches'].to_numpy()):
        targets = poi2ids[id2poi[id_str]]
        preds = set(matches.split())
        score = len((targets & preds)) / len((targets | preds))
        if score < 1:
            nonideal.append([id_str, targets, preds])
        scores.append(score)
    scores = np.array(scores)
    return scores.mean(), nonideal


def calculate_iou_ceil(data, train_data, id2poi, poi2ids, return_nonideal=False):
    eval_df = pd.DataFrame()
    eval_df['id'] = data['id'].unique().tolist()
    eval_df['match_id'] = eval_df['id']
    print('Unique id: %s' % len(eval_df))

    eval_df_ = train_data[train_data['label'] == 1][['id', 'match_id']]
    eval_df = pd.concat([eval_df, eval_df_])

    eval_df = eval_df.groupby('id')['match_id'].\
                            apply(list).reset_index()
    eval_df['matches'] = eval_df['match_id'].apply(lambda x: ' '.join(set(x)))
    print('Unique id: %s' % len(eval_df))

    if return_nonideal:
        iou_score, nonideal = get_score_and_nonideal(eval_df, id2poi, poi2ids)
        print('IoU score: %s' % iou_score)
        return nonideal
    iou_score = get_score(eval_df, id2poi, poi2ids)
    print('IoU score: %s' % iou_score)
    return iou_score


def split_folds(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    ## Data split
    kf = GroupKFold(n_splits=2)
    for i, (trn_idx, val_idx) in enumerate(kf.split(data,
                                                    data['point_of_interest'],
                                                    data['point_of_interest'])):
        data.loc[val_idx, 'set'] = i

    print('Num of train data: %s' % len(data))
    print(data['set'].value_counts())

    valid_data = data[data['set'] == 0]
    train_data = data[data['set'] == 1]

    print('Train data: ')
    analysis(train_data)
    print('Valid data: ')
    analysis(valid_data)

    train_poi = train_data['point_of_interest'].unique().tolist()
    valid_poi = valid_data['point_of_interest'].unique().tolist()

    print(set(train_poi) & set(valid_poi))

    train_ids = train_data['id'].unique().tolist()
    valid_ids = valid_data['id'].unique().tolist()

    print(set(train_ids) & set(valid_ids))

    tv_ids_d = {}
    tv_ids_d['train_ids'] = train_ids
    tv_ids_d['valid_ids'] = valid_ids

    del train_data, valid_data
    gc.collect()
    data = data.set_index('id')
    return data, tv_ids_d
