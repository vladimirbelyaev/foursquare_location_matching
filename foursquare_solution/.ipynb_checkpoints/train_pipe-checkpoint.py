import gc
import os
import random
import torch

import numpy as np
import pandas as pd
from foursquare_solution.features.featurizer import create_ds_with_feats

from foursquare_solution.raw_data_processor import (get_id2ids, get_id2poi,
                                                    get_poi2ids,
                                                    prepare_dataset)
from foursquare_solution.tools.common import seed_everything
from foursquare_solution.tools.embeddings import convert_to_embeds
from foursquare_solution.tools.train_analysis import (calculate_iou_ceil,
                                                      split_folds)

is_debug = False
SEED = 2022
data_root = './'
lang_model_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
use_cuda = torch.cuda.is_available()
num_splits = 3
num_splits_to_save_map = {
    'train': 3,
    'valid': 3
}


seed_everything(SEED)

data = pd.read_csv(os.path.join(data_root, 'train.csv'))
non_id_cols = [col for col in data.columns if 'id' not in col]
data.loc[:, non_id_cols] = data.loc[:, non_id_cols].applymap(
    lambda x: x.lower() if isinstance(x, str) else x
)

if is_debug:
    data = data.sample(n = 10000, random_state = SEED)
    data = data.reset_index(drop = True)


data, tv_ids_d = split_folds(data)
np.save('tv_ids_d.npy', tv_ids_d)

for mode in ['train', 'valid']:
    work_data = data.loc[tv_ids_d[f'{mode}_ids']].reset_index()
    vecs = convert_to_embeds(work_data.name.fillna('').tolist(), lang_model_name, use_cuda=use_cuda)
    # vecs = np.load(os.path.join(data_root, mode + '_embs.npy'))
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
    num_splits_to_save = num_splits_to_save_map[mode]
    for idx, ds_part in enumerate(create_ds_with_feats(
        work_data, train_data, vecs, tfidf_d, id2index_d, mode, num_splits, num_splits_to_save
    )):
        ds_part.to_parquet(f'{data_root}/{mode}_data{idx}.pqt')
    del work_data, train_data
    gc.collect()
