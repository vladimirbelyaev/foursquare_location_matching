from typing import Optional
import numpy as np
import pandas as pd

from foursquare_solution.recall_algos.neighbors import get_faiss_neighbors


def recall_faiss(
    df: pd.DataFrame, 
    vecs: np.ndarray, 
    Neighbors=10, 
    normalize=False, 
    tag='_name', 
    use_cuda: bool = False,
    batch_size: Optional[int] = None
):
    train_df = get_faiss_neighbors(
        df, 
        vecs, 
        base_neighbors=Neighbors, 
        tag=tag, 
        normalize=normalize, 
        use_cuda=use_cuda,
        batch_size=batch_size
    )
    return train_df


def recall_faiss_multi(df, vecs, coordinate_vec, coefs, neighbors = 20, base_tag='_name', use_cuda=True, batch_size=50000):
    vecs = vecs / (np.sqrt((vecs ** 2).sum(1))[:, None])
    res_df = None
    for coef in coefs:
        vecs_ = np.hstack([
            vecs, 
            coordinate_vec * coef
        ])
        df_vec = recall_faiss(
            df, vecs_, neighbors, normalize=False, 
            tag=base_tag + '_' + str(coef).replace('.', '_'), use_cuda=use_cuda, batch_size=batch_size
        )
        if res_df is None:
            res_df = df_vec
        else:
            res_df = res_df.merge(df_vec, on=['id', 'match_id'], how='outer')
    return res_df