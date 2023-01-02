
import gc
import time
from functools import partial

import cdifflib
import Levenshtein
import numpy as np
import pylcs

feat_columns = [
    'name', 'address', 'city',
    'state', 'zip', 'url',
    'phone', 'categories', 'country', #'city_geocoded'
]
vec_columns = [
    'name', 'categories', 'address', 'state', 'url', 'country',# 'city_geocoded'
]


smatcher = cdifflib.CSequenceMatcher()
def gesher(s, match_s):
    smatcher.set_seqs(s, match_s)
    return smatcher.ratio()


def collen(lst):
    return [len(x) if x != 'nan' else np.nan for x in lst]


def lstfunc(l1, l2, func):
    return [np.nan if a == 'nan' or b == 'nan' else func(a, b) for a, b in zip(l1, l2)]


gesher = partial(lstfunc, func=gesher)
levenser = partial(lstfunc, func=Levenshtein.distance)
jaroer = partial(lstfunc, func=Levenshtein.jaro_winkler)
lcser = partial(lstfunc, func=pylcs.lcs)


def calculate_feature(df, data, tfidf_d, id2index_d, col):
    print(time.time(), 'veccol multiplies')
    if col in vec_columns:
        tv_fit = tfidf_d[col]
        indexs = [id2index_d[i] for i in df['id']]
        match_indexs = [id2index_d[i] for i in df['match_id']]
        df[f'{col}_sim'] = tv_fit[indexs].multiply(tv_fit[match_indexs]).sum(axis = 1).A.ravel()
        del indexs, match_indexs
        gc.collect()

    col_values = data.loc[df['id']][col].values.astype(str).tolist()
    matcol_values = data.loc[df['match_id']][col].values.astype(str).tolist()
    gc.collect()

    print(time.time(), 'gesh')
    df[f'{col}_gesh'] = gesher(col_values, matcol_values)
    gc.collect()
    print(time.time(), 'leven')
    df[f'{col}_leven'] = levenser(col_values, matcol_values)
    gc.collect()
    print(time.time(), 'jaro')
    df[f'{col}_jaro'] = jaroer(col_values, matcol_values)
    gc.collect()
    print(time.time(), 'lcs')
    df[f'{col}_lcs'] = lcser(col_values, matcol_values)
    gc.collect()

    print(time.time(), 'finalize')
    if col not in ['phone', 'zip']:
        df[f'{col}_len'] = collen(col_values)
        df[f'match_{col}_len'] = collen(matcol_values)
        df[f'{col}_len_diff'] = np.abs(df[f'{col}_len'] - df[f'match_{col}_len'])
        df[f'{col}_nleven'] = df[f'{col}_leven'] / \
                                df[[f'{col}_len', f'match_{col}_len']].max(axis = 1)

        df[f'{col}_nlcsk'] = df[f'{col}_lcs'] / df[f'match_{col}_len']
        df[f'{col}_nlcs'] = df[f'{col}_lcs'] / df[f'{col}_len']

        df = df.drop(f'{col}_len', axis = 1)
        df = df.drop(f'match_{col}_len', axis = 1)
        gc.collect()
    print(time.time(), 'end')
    return df
