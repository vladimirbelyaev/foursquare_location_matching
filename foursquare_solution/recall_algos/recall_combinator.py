from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from foursquare_solution.recall_algos.location_wise import (recall_knn_basic,
                                                            recall_knn_country)
from foursquare_solution.recall_algos.simple import recall_simple
from foursquare_solution.recall_algos.vector_wise import recall_faiss, recall_faiss_multi


def recall_all(
    df: pd.DataFrame, 
    vecs: Optional[List[Tuple[np.ndarray, str, List[float]]]] = None,
    coo_cols=['longitude', 'latitude'],
    Neighbors=10,
    Neighbors_vec=10,
    use_cuda: bool = False,
    batch_size: Optional[int] = None
):
    df_all = recall_knn_basic(df, Neighbors, enhanced=True, convert_to_radians=False)
    coordinates = df[coo_cols].values
    if vecs is not None:
        for vec_arr, tag, coefs in vecs:
            df_vec = recall_faiss_multi(
                df, vec_arr, coordinates, 
                neighbors=Neighbors_vec,
                coefs=coefs,
                base_tag=tag, 
                use_cuda=use_cuda, 
                batch_size=batch_size
            )
            df_all = df_all.merge(df_vec, on=['id', 'match_id'], how='outer')
            del df_vec
    df_country = recall_knn_country(df, Neighbors, enhanced=True, convert_to_radians=True)
    df_all = df_all.merge(df_country, on=['id', 'match_id'], how='outer')
    del df_country
    df_simple = recall_simple(df, threshold=2)
    df_all = df_all.merge(df_simple, on=['id', 'match_id'], how='outer')
    return df_all
