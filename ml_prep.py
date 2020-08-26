import pickle
import re
import pandas as pd
import numpy as np
import fuzzywuzzy
from fuzzywuzzy import process

import warnings
warnings.filterwarnings('ignore')



def whole_to_ml(df):

    remove_strings = ['year', 'sitecode']
    remove_cols = _get_list(df, remove_strings)
    df = df.drop(remove_cols, axis = 1)

    # For making dummies of codeypes that have multiple columns
    code_types = ['scode']
    dummy_dfs = []
    for code_str in code_types:
        into_list = []
        into_list.append(code_str)
        dummies_cols = _get_list(df, into_list)
        dummies = []
        for col in dummies_cols:
            dummies.append(pd.get_dummies(df[col]))
        try:
            site_dummies = pd.concat(dummies, axis = 1)
            df = df.drop(dummies_cols, axis = 1)
            site_dummies = site_dummies.groupby(level=0, axis=1).sum()
            site_dummies = site_dummies.add_prefix(code_str + '=')
            dummy_dfs.append(site_dummies)
        except ValueError:
            print('no scode columns')

    restring_strings = ['code']
    restring_cols = _get_list(df, restring_strings)
    whole2 = pd.DataFrame({col:str(col)+'=' for col in df},
     index=df.index) + df.astype(str)
    df = df.drop(restring_cols, axis = 1)
    for col in restring_cols:
        df[col] = whole2[col]

    dummies_strings = ['code', 'bap']#, 'strat'
    dummies_cols = _get_list(df, dummies_strings)
    dummies = []
    for col in dummies_cols:
        dummies.append(pd.get_dummies(df[col]))
    site_dummies = pd.concat(dummies, axis = 1)
    df = df.drop(dummies_cols, axis = 1)
    df = pd.concat((df, site_dummies), axis = 1)

    # re attaching the dummy dfs that have multi column codes after
    # operations on 'code' columns
    try:
        for data_f in dummy_dfs:
            df = pd.concat((df, data_f), axis = 1)
    except:
        pass

    remove_strings = ['nan']
    remove_cols = _get_list(df, remove_strings)
    df = df.drop(remove_cols, axis = 1)

    return df
