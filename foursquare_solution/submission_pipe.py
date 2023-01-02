import gc
import torch

import numpy as np
import pandas as pd
import catboost as cb
from foursquare_solution.features.featurizer import create_ds_with_feats
from foursquare_solution.models.prediction_grouping import to_prediction

from foursquare_solution.raw_data_processor import (get_id2ids, get_id2poi,
                                                    get_poi2ids,
                                                    prepare_dataset)
from foursquare_solution.tools.common import seed_everything
from foursquare_solution.tools.embeddings import convert_to_embeds
from foursquare_solution.tools.train_analysis import (calculate_iou_ceil,
                                                      split_folds)
from foursquare_solution.models.features import train_cols

is_debug = False
SEED = 2022
data_root = '/kaggle/input/foursquare-location-matching'
lang_model_name = '/kaggle/input/multiqaminilml6cosv1/multi-qa-MiniLM-L6-cos-v1/'
cb_model_path = '/kaggle/input/cb-foursquare-model/cb_model'
mode = 'eval'
use_cuda = torch.cuda.is_available()
num_splits = 11
seed_everything(SEED)

data = pd.read_csv(f'{data_root}/test.csv')
if len(data) < 20:
    data = pd.read_csv(f'{data_root}/train.csv', nrows = 100)
    data = data.drop('point_of_interest', axis = 1)

non_id_cols = [col for col in data.columns if 'id' not in col]
data.loc[:, non_id_cols] = data.loc[:, non_id_cols].applymap(
    lambda x: x.lower() if isinstance(x, str) else x
)


work_data = data
vecs = convert_to_embeds(work_data.name.fillna('').tolist(), lang_model_name, use_cuda=use_cuda)
print(vecs.shape, vecs.sum())
train_data, tfidf_d, id2index_d = prepare_dataset(
    work_data, vecs, num_neighbors=20, is_train_mode=True
)

id2poi = get_id2poi(work_data)
poi2ids = get_poi2ids(work_data)
id2ids = get_id2ids(id2poi, poi2ids)
np.save(f'id2ids_{mode}', id2ids, allow_pickle=True)
del id2ids

calculate_iou_ceil(work_data, train_data, id2poi, poi2ids, True)
gc.collect()
preds = []
for ds_part in create_ds_with_feats(
    work_data, train_data, vecs, tfidf_d, id2index_d, mode, num_splits, num_splits
):
    model = cb.CatBoostClassifier().load_model(cb_model_path)
    ds_part['prediction'] = model.predict_proba(ds_part[train_cols])
    prediction_df = to_prediction(ds_part)
    preds.append(prediction_df)
    del model, ds_part
    gc.collect()
