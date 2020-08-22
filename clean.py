import pickle
import re
import pandas as pd
import numpy as np
import fuzzywuzzy
from fuzzywuzzy import process

import warnings
warnings.filterwarnings('ignore')


def _get_list(df, list_names, not_list = []):

    strings = list_names
    choose_cols = []
    df_columns = df.columns.tolist()
    for col in df_columns:
        for r_str in strings:
            if r_str.lower() in col.lower() :
                choose_cols.append(col)

    res = []
    for choice in choose_cols:
        res.append(choice)
        for nl in not_list:
            if nl.lower() in choice.lower():
                del res[-1]

    return res

def _destring_plot_id(df):

    for i in df['PLOT_ID'].iteritems():
        if isinstance(i[1], str):
            if 'control' in i[1].lower():
                num_list = [int(nn) for nn in i[1].split() if nn.isdigit()]
                df.loc[i[0], 'PLOT_ID'] = num_list[-1] + 100
            df.loc[i[0], 'PLOT_ID'] = re.sub("[^0-9]", "", df.loc[i[0],
                                                             'PLOT_ID'])

    df["PLOT_ID"] = df["PLOT_ID"].astype(float)

def whole_clean(df):

    df = df[df['PLOT_ID'].notna()]
    df = df.dropna(how='all', axis = 1)
    df = df.dropna(how='all', axis = 0)
    df = df.rename(columns=lambda x: x.strip())
    # Make all the column titles lower case as there is variation amonst sites
    df.columns = df.columns.str.lower()

    # Dealing with plot id to amke each plot unique for each survey
    df["plot_id"] = df["plot_id"].astype(str)
    year_str = str(int(df['year'][0]))
    df['index_id'] = df['plot_id'] + '_' + df['sitecode'] + '_' + year_str

    remove_strings = ['note', 'comment', 'remark', 'date', 'data_issue',
        'qa', 'bng_grid', 'survey', 'nvc', 'strat']
    remove_cols = _get_list(df, remove_strings)
    df = df.drop(remove_cols, axis = 1)

    df = df.replace(r'^\s*$', np.NaN, regex=True)

    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    destring_strings = ['landuse', 'slopeform', 'scode', 'slope',
     'aspect', 'altitude']
    destring_cols = _get_list(df, destring_strings)
    for col in destring_cols:
        for i in df[col].iteritems():
            if isinstance(i[1], str):
                if any(map(str.isdigit, i[1])):
                    df.loc[i[0], col] = re.search(r'\d+', i[1]).group()
                else:
                    df.loc[i[0], col] = np.NaN
        df[col] = df[col].astype(float)

    median_str = ['aspect', 'altitude', 'eastings', 'northings', 'slope']
    median_cols = _get_list(df, median_str, not_list=['slopeform'])
    for col in median_cols:
        df[col] = df[col].fillna(df[col].median())

    df = df.set_index('index_id')

    # Make all the column titles lower case as there is variation amonst sites
    df.columns = df.columns.str.lower()

    return df

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

def species_clean(df):

    df = df[df['PLOT_ID'].notna()]
    df = df.dropna(how='all', axis = 1)
    df = df.dropna(how='all', axis = 0)
    df = df.rename(columns=lambda x: x.strip())
    # Make all the column titles lower case as there is variation amonst sites
    df.columns = df.columns.str.lower()

    # Dealing with plot id to amke each plot unique for each survey
    df["plot_id"] = df["plot_id"].astype(str)
    year_str = str(int(df['year'][0]))
    df['index_id'] = df['plot_id'] + '_' + df['sitecode'] + '_' + year_str

    df = df.replace(r'^\s*$', np.NaN, regex=True)

    df = df[df['desc_latin'] != 'Unidentified']

    remove_strings = ['data', 'comment', 'qa', 'cell', 'species_no']#'scode',
    remove_cols = _get_list(df, remove_strings)
    df = df.drop(remove_cols, axis = 1)

    destring_strings = ['percent', 'freq']
    destring_cols = _get_list(df, destring_strings)
    for col in destring_cols:
        for i in df[col].iteritems():
            if isinstance(i[1], str):
                if any(map(str.isdigit, i[1])):
                    df.loc[i[0], col] = re.search(r'\d+', i[1]).group()
                else:
                    df.loc[i[0], col] = np.NaN
        df[col] = df[col].astype(float)

    median_str = ['percent', 'frequency']
    median_cols = _get_list(df, median_str)
    for col in median_cols:
        df[col] = df[col].fillna(df[col].median())

    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    df = df.dropna()

    # Make all the column titles lower case as there is variation amonst sites
    df.columns = df.columns.str.lower()

    return df

def get_abund_and_freq(df, column):

    df_dict = {num: df.loc[df['index_id'] == num] for num in df['index_id']}
    cover_plots = []
    frequency_plots = []
    for key in df_dict.keys():
        df_dict[key] = df_dict[key].drop_duplicates(subset=[column],
                                                          keep='first')

        #print(df_dict[key].head())
        abund_inst = pd.DataFrame()
        abund_inst = df_dict[key].pivot(index = 'index_id', columns = column,
                                                     values = 'percent_cover')
        cover_plots.append(abund_inst)

        freq_inst = pd.DataFrame()
        freq_inst = df_dict[key].pivot(index = 'index_id', columns = column,
                                                         values = 'frequency')
        frequency_plots.append(freq_inst)

    df_abund = pd.concat(cover_plots, axis=1)
    df_abund = df_abund.groupby(level=0, axis=1).sum()
    #df_abund = df_abund.reset_index()
    df_freq = pd.concat(frequency_plots, axis=1)
    df_freq = df_freq.groupby(level=0, axis=1).sum()
    #df_freq = df_freq.reset_index()

    return df_abund, df_freq

def ground_clean(df):

    df = df[df['PLOT_ID'].notna()]
    df = df.dropna(how='all', axis = 1)
    df = df.dropna(how='all', axis = 0)
    df = df.rename(columns=lambda x: x.strip())
    # Make all the column titles lower case as there is variation amonst sites
    df.columns = df.columns.str.lower()

    # Dealing with plot id to amke each plot unique for each survey
    df["plot_id"] = df["plot_id"].astype(str)
    year_str = str(int(df['year'][0]))
    df['index_id'] = df['plot_id'] + '_' + df['sitecode'] + '_' + year_str

    # Some of the plots dont have frequency and only have percent cover,
    # frequency values are typically about a quarter of the percent (out of 25)
    if 'frequency' not in df:
        df['frequency'] = df['percent_cover'] / 4
        print('\n\nplot ' + df['sitecode'] + df['year'] + ' with no frequency')

    # taking the trailing and leading spaces from the features
    df['feature'] = df['feature'].apply(lambda x: x.strip())

    df_other = df[df['feature'].str.lower() != 'vegetation height']

    g_cover, g_frequency = get_abund_and_freq(df_other, column='feature')

    df_height = df[df['feature'].str.lower() == 'vegetation height']

    destring_cols = _get_list(df_height, ['cell'])
    for col in destring_cols:
        if df_height[col].dtype == object:
            df_height[col] = df_height[col].replace({"<1m": 100.0,
                                '>100': 100.0,
                                 "*": np.NaN,
                                 'xz': np.NaN,
                                 'DW': np.NaN,
                                 'dw': np.NaN,
                                 '.': np.NaN,
                                 '27.5.': 27.5,
                                 '`13': 13.0,
                                 '>55': 55.0,
                                 'Tree': 100.0,
                                 'Tree 1': 100.0,
                                 '4..5': 4.5,
                                 '7,5': 7.5,
                                 'Hole': np.NaN})
            df_height[col] = df_height[col].astype(float)
        else:
            pass

    try:
        df_height['max_height'] = df_height.loc[:, 'cell_1':'cell_25'].max(1)
        df_height['median_height'] =\
                                 df_height.loc[:, 'cell_1':'cell_25'].median(1)
    except:
        df_height['max_height'] = np.NaN
        df_height['median_height'] = np.NaN


    median_str = ['median_height', 'max_height']
    for col in median_str:
        df_height[col] = df_height[col].fillna(df_height[col].median())

    df_height = df_height[df_height.duplicated('index_id',
                                                        keep='first') == False]

    remove_strings = ['percent', 'freq', 'data', 'cell', 'feature', 'site',
        'code', 'year', 'qa']
    remove_cols = _get_list(df_height, remove_strings)
    df_height = df_height.drop(remove_cols, axis = 1)

    df_base = pd.DataFrame()
    base_strings = ['sitec', 'mcode', 'index']
    base_cols = _get_list(df, base_strings)
    for col in base_cols:
        df_base[col] = df[col]
    df_base = df_base[df_base.duplicated('index_id', keep='first') == False]

    df_base = df_base.set_index('index_id')
    df_height = df_height.set_index('index_id')
    #g_cover = g_cover.set_index('index')
    #g_frequency = g_frequency.set_index('index')

    g_cover = g_cover.add_prefix('cover-')
    g_frequency = g_frequency.add_prefix('freq-')

    df_final = pd.concat((df_base, df_height, g_frequency, g_cover), axis = 1)

    #df_final['index_id'] = df_final.index

    median_cols = _get_list(df_final, ['height'])
    for col in median_cols:
        df_final[col] = df_final[col].fillna(df_final[col].median())
    zero_cols = _get_list(df_final, ['freq', 'cover'])
    for col in zero_cols:
        df_final[col] = df_final[col].fillna(0)

    # Make all the column titles lower case as there is variation amonst sites
    df_final.columns = df_final.columns.str.lower()

    return df_final

