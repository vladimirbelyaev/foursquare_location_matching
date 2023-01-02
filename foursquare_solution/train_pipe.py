from collections import Counter
import gc
import json
import os
import pickle
import torch
import h3

import numpy as np
import pandas as pd
from unidecode import unidecode
from foursquare_solution.features.agg_features import avg_haversine_country, avg_neighbors_country
from foursquare_solution.features.featurizer import create_ds_with_feats

from foursquare_solution.raw_data_processor import (get_id2ids, get_id2poi,
                                                    get_poi2ids,
                                                    prepare_dataset)
from foursquare_solution.tools.common import seed_everything
from foursquare_solution.tools.embeddings import convert_to_embeds
from foursquare_solution.tools.train_analysis import (calculate_iou_ceil,
                                                      split_folds)


is_debug = False
use_cuda = torch.cuda.is_available()
data_root = './'
save_data_root = './train_data_githubbed'
num_splits = 7
num_splits_to_save_map = {
    'train': 7,
    'valid': 7
}
faiss_batch_size = 50000

SEED = 2022
lang_model_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
decode_name = True
strip_phone = False
save_embeds = False
load_embeds = False

seed_everything(SEED)

if not os.path.exists(save_data_root):
    print("creating directory for output data:", save_data_root)
    os.mkdir(save_data_root)

data = pd.read_csv(os.path.join(data_root, 'train.csv'))

# name - unidecoding KR and TH only
if decode_name:
    data.loc[:, 'name'] = data.loc[:, 'name'].fillna('null')
    is_kr_th = data.country.isin(set(['KR', 'TH'])).values
    data.loc[:, 'name'] = (data.loc[:, 'name'].apply(unidecode).str.strip() * is_kr_th) + (data.loc[:, 'name'] * (~is_kr_th))

# phone
if strip_phone:
    import re
    non_numeric = re.compile("[^0-9]")
    data.loc[:, 'phone'] = data.loc[:, 'phone'].apply(lambda x: non_numeric.sub('', x)[-7:] if not isinstance(x, float) else x)

non_id_cols = [col for col in data.columns if 'id' not in col]
data.loc[:, non_id_cols] = data.loc[:, non_id_cols].applymap(
    lambda x: x.lower() if isinstance(x, str) else x
)

if is_debug:
    data = data.sample(n = 10000, random_state = SEED)
    data = data.reset_index(drop = True)


data, tv_ids_d = split_folds(data)
np.save(f'{save_data_root}/tv_ids_d.npy', tv_ids_d)

avg_nbs_all_spl = None
avg_hvs_all_spl = None

for mode in ['train', 'valid']:
    work_data = data.loc[tv_ids_d[f'{mode}_ids']].reset_index()
    vecs = convert_to_embeds(work_data.name.fillna('').tolist(), lang_model_name, use_cuda=use_cuda)
    # vecs = np.load(os.path.join(data_root, mode + '_embs.npy'))
    print(vecs.shape, vecs.sum())
    train_data, tfidf_d, id2index_d = prepare_dataset(
        work_data, vecs, num_neighbors=20, num_neighbors_vec=15, is_train_mode=True, use_cuda=True, batch_size=50000
    )
    with open(f'{save_data_root}/id2index_d_{mode}.pkl', 'wb') as f:
        pickle.dump(id2index_d, f)
    with open(f'{save_data_root}/tfidf_d_{mode}.pkl', 'wb') as f:
        pickle.dump(tfidf_d, f)

    id2poi = get_id2poi(work_data)
    poi2ids = get_poi2ids(work_data)
    id2ids = get_id2ids(id2poi, poi2ids)
    np.save(f'{save_data_root}/id2ids_{mode}', id2ids, allow_pickle=True)
    del id2ids

    calculate_iou_ceil(work_data, train_data, id2poi, poi2ids, True)
    gc.collect()
    avg_hvs = avg_haversine_country(work_data)
    with open(f'{save_data_root}/avg_hvs_{mode}.pkl', 'wb') as f:
        pickle.dump(avg_hvs, f)
    avg_nbs = avg_neighbors_country(work_data)
    with open(f'{save_data_root}/avg_nbs_{mode}.pkl', 'wb') as f:
        pickle.dump(avg_nbs, f)
    if mode == 'train':
        avg_hvs_all_spl = avg_hvs
        avg_nbs_all_spl = avg_nbs
        
    
    h3_counts = Counter([h3.geo_to_h3(lat, lng, 7) for lat, lng in zip(data['latitude'], data['longitude'])])
    with open(f'{save_data_root}/h3_7_{mode}.json', 'w') as f:
        json.dump(h3_counts, f)
    
    
    num_splits_to_save = num_splits_to_save_map[mode]
    for idx, ds_part in enumerate(create_ds_with_feats(
        work_data, train_data, vecs, tfidf_d, id2index_d, avg_hvs_all_spl, avg_nbs_all_spl, h3_counts,
        mode, num_splits, num_splits_to_save
    )):
        ds_part.to_parquet(f'{save_data_root}/{mode}_data{idx}.pqt')
    del work_data, train_data
    gc.collect()
