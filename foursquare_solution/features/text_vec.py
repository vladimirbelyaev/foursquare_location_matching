import numpy as np
from tqdm.auto import tqdm

from foursquare_solution.tools.types_downcasting import downcast_floats


def text_distance_cached(df, data, chunk_size=100000):
    scores_lst = []
    scores_lst_cosine = []
    chunks_num = (df.shape[0] - 1) // chunk_size + 1
    for i in tqdm(range(chunks_num)):
        vec1 = data.loc[df['id'].iloc[chunk_size * i: chunk_size * (i + 1)]]['name_vec'].values
        vec2 = data.loc[df['match_id'].iloc[chunk_size * i: chunk_size * (i + 1)]]['name_vec'].values
        vec1_arr, vec2_arr = np.vstack(vec1), np.vstack(vec2)
        scores = (vec1_arr * vec2_arr).sum(1)
        scores_lst.append(scores)
        vec1_arr = vec1_arr / (np.sqrt((vec1_arr ** 2).sum(1))[:, None])
        vec2_arr = vec2_arr / (np.sqrt((vec2_arr ** 2).sum(1))[:, None])
        scores_cosine = (vec1_arr * vec2_arr).sum(1)
        scores_lst_cosine.append(scores_cosine)
    #data.drop(columns=['name_vec'], inplace=True)
    res = np.hstack(scores_lst)
    res_cosine = np.hstack(scores_lst_cosine)
    return res, res_cosine



def text_distance_from_vec(df, data, vecs, chunk_size=100000):
    data['indices'] = np.arange(data.shape[0])
    scores_lst = []
    scores_lst_cosine = []
    chunks_num = (df.shape[0] - 1) // chunk_size + 1
    for i in tqdm(range(chunks_num)):
        vec1_idxs = data.loc[df['id'].iloc[chunk_size * i: chunk_size * (i + 1)]]['indices'].values
        vec2_idxs = data.loc[df['match_id'].iloc[chunk_size * i: chunk_size * (i + 1)]]['indices'].values
        vec1_arr, vec2_arr = vecs[vec1_idxs, :], vecs[vec2_idxs, :]
        scores = (vec1_arr * vec2_arr).sum(1)
        scores_lst.append(scores)
        vec1_arr = vec1_arr / (np.sqrt((vec1_arr ** 2).sum(1))[:, None])
        vec2_arr = vec2_arr / (np.sqrt((vec2_arr ** 2).sum(1))[:, None])
        scores_cosine = (vec1_arr * vec2_arr).sum(1)
        scores_lst_cosine.append(scores_cosine)
    #data.drop(columns=['name_vec'], inplace=True)
    res = np.hstack(scores_lst)
    res_cosine = np.hstack(scores_lst_cosine)
    return res, res_cosine



def recalculate_text_distances(df, data, vecs):
    text_dist, text_dist_cosine = text_distance_from_vec(df, data, vecs)
    df['name_dist'] = text_dist
    df['name_dist_cos'] = text_dist_cosine
    downcast_floats(df)
    return df