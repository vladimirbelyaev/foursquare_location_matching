from typing import Optional, Type, Union
from tqdm.auto import tqdm

import faiss
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors


def get_basic_neighbors(
    df: pd.DataFrame, vecs: np.ndarray, neighbors=10, metric='minkowski', tag=''
):
    train_df = []
    knn = NearestNeighbors(n_neighbors=neighbors, n_jobs=-1, metric=metric)
    knn.fit(vecs, df.index)
    dists, nears = knn.kneighbors(vecs)

    for k in range(neighbors):
        cur_df = df[['id']]
        cur_df['match_id'] = df['id'].values[nears[:, k]]
        cur_df['kdist' + tag] = dists[:, k]
        cur_df['kneighbors' + tag] = k
        cur_df['found' + tag] = True
        train_df.append(cur_df)
    return pd.concat(train_df)


def get_kdtree_neighbors(
    data: pd.DataFrame,
    vecs: np.ndarray,
    tree_class: Union[Type[KDTree], Type[BallTree]],
    base_neighbors=20,
    metric='minkowski',
    tag='',
    max_threshold=None
) -> pd.DataFrame:
    tree = tree_class(vecs, metric=metric)
    dists1, neighbors1 = tree.query(vecs, k=base_neighbors)
    last_radius_found = dists1[:, -1]
    # query_radiuses = last_radius_found * (base_neighbors + 1) / (base_neighbors)
    # if max_threshold is not None:
    #     query_radiuses = np.minimum(query_radiuses, max_threshold)
    query_addition = last_radius_found / base_neighbors
    if max_threshold is not None:
        query_addition = np.minimum(query_addition, max_threshold)
    query_radiuses = last_radius_found + query_addition
    neighbors, dists = tree.query_radius(
        vecs, query_radiuses, return_distance=True, sort_results=True
    )
    idxs = np.hstack([np.ones_like(neighbors_) * idx for idx, neighbors_ in enumerate(neighbors)])
    kneighs = np.hstack([np.arange(dists_.shape[0]) for dists_ in dists])
    dists = np.hstack(dists)
    neighbors = np.hstack(neighbors)
    cur_df = pd.DataFrame(data['id'].values[idxs], columns=['id'])
    cur_df['match_id'] = data['id'].values[neighbors]
    cur_df['kdist' + tag] = dists
    cur_df['kneighbors' + tag] = kneighs
    cur_df['found' + tag] = True
    return cur_df


class FaissIndex:
    def __init__(self, vecs: np.ndarray):
        self._dim = vecs.shape[1]
        self._nlist = int(vecs.shape[0] ** 0.5)
        self._quantizer = faiss.IndexFlatL2(self._dim)
        self.index = faiss.IndexIVFFlat(self._quantizer, self._dim, self._nlist)


def get_faiss_neighbors_(
    df: pd.DataFrame, vecs: np.ndarray, base_neighbors=20, tag='', normalize=False
):
    if normalize:
        vecs = vecs / (np.sqrt((vecs ** 2).sum(1))[:, None])
    dim = vecs.shape[1]
    nlist = int(vecs.shape[0] ** 0.5)
    #index = faiss.IndexFlatL2(dim)
    vecs = vecs.astype(np.float32).copy(order='C')
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index.nprobe = 50
    print('train index', vecs.shape)
    index.train(vecs)
    print('add to index')
    index.add(vecs)
    print('search in index')
    dists, nears = index.search(vecs, base_neighbors)
    train_df = []

    for k in range(base_neighbors):
        cur_df = df[['id']]
        cur_df['match_id'] = df['id'].values[nears[:, k]]
        cur_df['kdist' + tag] = dists[:, k]
        cur_df['kneighbors' + tag] = k
        cur_df['found' + tag] = True
        train_df.append(cur_df)
    return pd.concat(train_df)



def get_faiss_neighbors(
    df: pd.DataFrame, 
    vecs: np.ndarray, 
    base_neighbors=20, 
    tag='', 
    normalize: bool = False,
    use_cuda: bool = False,
    batch_size: Optional[int] = None 
):
    if normalize:
        vecs = vecs / (np.sqrt((vecs ** 2).sum(1))[:, None])
    vecs = vecs.astype(np.float32).copy(order='C')
    index = FaissIndex(vecs).index
    index.nprobe = 50
    if use_cuda:
        res = faiss.StandardGpuResources()
        print('moving to gpu')
        index = faiss.index_cpu_to_gpu(res, 0, index)
    print('train index', vecs.shape)
    index.train(vecs)
    print('add to index')
    index.add(vecs)
    print('search in index')
    if batch_size is None:
        dists, nears = index.search(vecs, base_neighbors)
    else:
        num_batches = (vecs.shape[0] - 1) // batch_size + 1
        dists_lst, nears_lst = [], []
        for i in tqdm(range(num_batches)):
            dists, nears = index.search(
                vecs[batch_size * i:batch_size * (i + 1), :], base_neighbors
            )
            dists_lst.append(dists)
            nears_lst.append(nears)
        nears = np.vstack(nears_lst)
        dists = np.vstack(dists_lst)
        del nears_lst, dists_lst
    train_df = []

    for k in range(base_neighbors):
        cur_df = df[['id']]
        cur_df['match_id'] = df['id'].values[nears[:, k]]
        cur_df['kdist' + tag] = dists[:, k]
        cur_df['kneighbors' + tag] = k
        cur_df['found' + tag] = True
        cur_df = cur_df[nears[:, k] != -1]
        train_df.append(cur_df)
    return pd.concat(train_df)