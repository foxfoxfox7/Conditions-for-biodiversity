import pickle
import re
import pandas as pd
import numpy as np
import fuzzywuzzy
from fuzzywuzzy import process

import warnings
warnings.filterwarnings('ignore')


def _get_list(df, list_names, not_list = []):
    '''
    Looks for any column names with the strings provided in them
    Also drops any with the string in not_list in them
    Used so that exact names are needed in case of typos
    '''

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

########################################################################
# initial cleaning
########################################################################

def whole_clean(df):
    '''
    Cleans the 'whole plot data' sheet in each survey excel file
    '''

    df = df[df['PLOT_ID'].notna()]
    df = df.dropna(how='all', axis = 1)
    df = df.dropna(how='all', axis = 0)
    df = df.rename(columns=lambda x: x.strip())
    # Make all the column titles lower case as there is variation amonst sites
    df.columns = df.columns.str.lower()

    # Some of the columns that are the same value have missing values
    year = df['year'].loc[df['year'].first_valid_index()]
    sitecode = df['sitecode'].loc[df['sitecode'].first_valid_index()]
    mcode = df['mcode'].loc[df['sitecode'].first_valid_index()]
    df['year'] = df['year'].fillna(year)
    df['sitecode'] = df['sitecode'].fillna(sitecode)
    df['mcode'] = df['mcode'].fillna(mcode)

    # some of the years are written as floats
    df['year'] = df['year'].astype(int)
    # some of the inputs for light are typos making them strings
    if 'light' in df.columns:
        for i in df['light'].iteritems():
            if isinstance(i[1], str):
                df.loc[i[0], 'light'] = np.NaN
        df['light'] = df['light'].astype(float)

    # Dealing with plot id to amke each plot unique for each survey
    # some plot_id are float, some are int some are object
    if df["plot_id"].dtype == float:
        df["plot_id"] = df["plot_id"].astype(int)
    df["plot_id"] = df["plot_id"].astype(str)

    # there are some leading and trailing spaces in the plot_id strings
    df['plot_id'] = df['plot_id'].str.strip()
    df['sitecode'] = df['sitecode'].str.strip()

    year_str = str(int(df['year'][0]))
    df['index_id'] = df['plot_id'] + '_' + df['sitecode'] + '_' + year_str

    # removing a load of columns that have no useful information
    remove_strings = ['note', 'comment', 'remark', 'date', 'data_issue',
        'qa', 'bng_grid', 'survey', 'strat']
    remove_cols = _get_list(df, remove_strings)
    df = df.drop(remove_cols, axis = 1)

    # replaces blank strings with NaN ie ''
    df = df.replace(r'^\s*$', np.NaN, regex=True)

    # strips leading and trailing spaces from any columns that have strings
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    # These columns shoudl be floats but because of typos are sometimes
    # strings. looks for typos to correct and converts to float
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

    # All columns with numerical data. The values are typical so median
    # is usually not a bad guess
    median_str = ['aspect', 'altitude', 'eastings', 'northings', 'slope']
    median_cols = _get_list(df, median_str, not_list=['slopeform'])
    for col in median_cols:
        df[col] = df[col].fillna(df[col].median())

    df = df.set_index('index_id')

    # Make all the column titles lower case as there is variation amonst sites
    df.columns = df.columns.str.lower()

    return df

def species_clean(df):
    '''
    Cleans the 'species template' sheet in each survey excel file
    '''

    # only takes the rows that have values in 'PLOT_ID'
    # There are many blank rows, often at the end of the sheet
    df = df[df['PLOT_ID'].notna()]
    # drops any columsn or rows that are all NaNs
    df = df.dropna(how='all', axis = 1)
    df = df.dropna(how='all', axis = 0)
    # strips leading and trailing spaces from the colum titles
    df = df.rename(columns=lambda x: x.strip())
    # Make all the column titles lower case as there is variation amonst sites
    df.columns = df.columns.str.lower()

    # Some of the columns that are the same value have missing values
    year = df['year'].loc[df['year'].first_valid_index()]
    sitecode = df['sitecode'].loc[df['sitecode'].first_valid_index()]
    mcode = df['mcode'].loc[df['sitecode'].first_valid_index()]
    df['year'] = df['year'].fillna(year)
    df['sitecode'] = df['sitecode'].fillna(sitecode)
    df['mcode'] = df['mcode'].fillna(mcode)

    # some of the years are written as floats
    df['year'] = df['year'].astype(int)

    # Dealing with plot id to amke each plot unique for each survey
    # some plot_id are float, some are int some are object
    if df["plot_id"].dtype == float:
        df["plot_id"] = df["plot_id"].astype(int)
    df["plot_id"] = df["plot_id"].astype(str)

    # there are some leading and trailing spaces in the plot_id strings
    df['plot_id'] = df['plot_id'].str.strip()
    df['sitecode'] = df['sitecode'].str.strip()

    year_str = str(int(df['year'][0]))
    df['index_id'] = df['plot_id'] + '_' + df['sitecode'] + '_' + year_str

    df = df.replace(r'^\s*$', np.NaN, regex=True)

    df = df[df['desc_latin'] != 'Unidentified']

    remove_strings = ['data', 'comment', 'qa', 'cell', 'species_no']#'scode',
    remove_cols = _get_list(df, remove_strings)
    df = df.drop(remove_cols, axis = 1)

    # there are some strings in the percent cover which need to removed
    for i in df['percent_cover'].iteritems():
        if isinstance(i[1], str):
            df.loc[i[0], 'percent_cover'] = np.NaN
    df['percent_cover'] = df['percent_cover'].astype(float)

    # strips leading and trailing spaces from any columns that have strings
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

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

    # All columns with numerical data. The values are approximatesly
    # gaussian distributed so median not a bad guess
    median_str = ['percent', 'frequency']
    median_cols = _get_list(df, median_str)
    for col in median_cols:
        df[col] = df[col].fillna(df[col].median())

    # drops any rows that have NaNs in them
    df = df.dropna()

    # Make all the column titles lower case as there is variation amonst sites
    df.columns = df.columns.str.lower()

    return df

def get_abund_and_freq(df, column):
    '''
    makes pivot tables out of the data. Used when the data has more than
    one row per plot. this is true on some of the sheets so needs to be
    converted to a different form to match with 'whole plot data'
    '''

    # making a dictionary for each plot id
    df_dict = {num: df.loc[df['index_id'] == num] for num in df['index_id']}
    cover_plots = []
    frequency_plots = []
    # splitting the DF into one for each plot before transforming and combining
    for key in df_dict.keys():
        df_dict[key] = df_dict[key].drop_duplicates(subset=[column],
                                                          keep='first')

        abund_inst = pd.DataFrame()
        abund_inst = df_dict[key].pivot(index = 'index_id', columns = column,
                                                     values = 'percent_cover')
        cover_plots.append(abund_inst)

        freq_inst = pd.DataFrame()
        freq_inst = df_dict[key].pivot(index = 'index_id', columns = column,
                                                         values = 'frequency')
        frequency_plots.append(freq_inst)

    # combining the reulting pivot tables for each plot into a survey DF
    df_abund = pd.concat(cover_plots, axis=1)
    df_abund = df_abund.groupby(level=0, axis=1).sum()
    df_freq = pd.concat(frequency_plots, axis=1)
    df_freq = df_freq.groupby(level=0, axis=1).sum()

    return df_abund, df_freq

def ground_clean(df):
    '''
    Cleans the 'ground features' sheet in each survey excel file
    '''

    df = df[df['PLOT_ID'].notna()]
    df = df.dropna(how='all', axis = 1)
    df = df.dropna(how='all', axis = 0)
    df = df.rename(columns=lambda x: x.strip())
    # Make all the column titles lower case as there is variation amonst sites
    df.columns = df.columns.str.lower()

    # Some of the columns that are the same value have missing values
    year = df['year'].loc[df['year'].first_valid_index()]
    sitecode = df['sitecode'].loc[df['sitecode'].first_valid_index()]
    mcode = df['mcode'].loc[df['sitecode'].first_valid_index()]
    df['year'] = df['year'].fillna(year)
    df['sitecode'] = df['sitecode'].fillna(sitecode)
    df['mcode'] = df['mcode'].fillna(mcode)

    # some of the years are written as floats
    df['year'] = df['year'].astype(int)

    # Dealing with plot id to amke each plot unique for each survey
    # some plot_id are float, some are int, some are object
    if df["plot_id"].dtype == float:
        df["plot_id"] = df["plot_id"].astype(int)
    df["plot_id"] = df["plot_id"].astype(str)

    # there are some leading and trailing spaces in the plot_id strings
    df['plot_id'] = df['plot_id'].str.strip()
    df['sitecode'] = df['sitecode'].str.strip()

    year_str = str(int(df['year'][0]))
    df['index_id'] = df['plot_id'] + '_' + df['sitecode'] + '_' + year_str

    # Some of the plots dont have frequency and only have percent cover,
    # frequency values are typically about a quarter of the percent (out of 25)
    if 'frequency' not in df:
        df['frequency'] = df['percent_cover'] / 4
        print('\n\nplot ' + str(df['sitecode'][0]) + str(df['year'][0])
         + ' with no frequency')

    # taking the trailing and leading spaces from the features
    df['feature'] = df['feature'].apply(lambda x: x.strip())
    # changing all feature labels to lower case to cover for discrepancies
    df['feature'] = df['feature'].str.lower()

    # there is 1 veg height for each plot so is taken as the DF base
    df_height = df[df['feature'].str.lower() == 'vegetation height']
    # The other catagories need to be transformed into a pivot table
    df_other = df[df['feature'].str.lower() != 'vegetation height']
    # Function which generates pivot tables
    # Used if there are multiple entries for each plot.1 plot should be 1 row
    g_cover, g_frequency = get_abund_and_freq(df_other, column='feature')

    # Correcting typos and alt inputs in the cell columns
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


    # reducing the cell columsn to just a median and max height
    try:
        df_height['max_height'] = df_height.loc[:, 'cell_1':'cell_25'].max(1)
        df_height['median_height'] =\
                                 df_height.loc[:, 'cell_1':'cell_25'].median(1)
    # There are some plots with no data here
    except:
        df_height['max_height'] = np.NaN
        df_height['median_height'] = np.NaN

    # There are some plots that have multiple entries This drops al but the 1st
    df_height = df_height[df_height.duplicated('index_id',
                                                        keep='first') == False]

    # remove a load of useless columns. done late as 'cell' is needed before
    remove_strings = ['percent', 'freq', 'data', 'cell', 'feature', 'site',
        'code', 'year', 'qa']
    remove_cols = _get_list(df_height, remove_strings)
    df_height = df_height.drop(remove_cols, axis = 1)

    # keeping only the essential information to add our transformed DFs to
    df_base = pd.DataFrame()
    base_strings = ['sitec', 'mcode', 'index']
    base_cols = _get_list(df, base_strings)
    for col in base_cols:
        df_base[col] = df[col]
    df_base = df_base[df_base.duplicated('index_id', keep='first') == False]

    # Setting the same index before combining the DFs
    # index for cover and frequency are already set during pivot table
    df_base = df_base.set_index('index_id')
    df_height = df_height.set_index('index_id')

    # labelling the column titles separately for when they need to be combined
    g_cover = g_cover.add_prefix('cover-')
    g_frequency = g_frequency.add_prefix('freq-')

    df_final = pd.concat((df_base, df_height, g_frequency, g_cover), axis = 1)

    # Filling NaNs
    median_cols = _get_list(df_final, ['height'])
    for col in median_cols:
        df_final[col] = df_final[col].fillna(df_final[col].median())
    zero_cols = _get_list(df_final, ['freq', 'cover'])
    for col in zero_cols:
        df_final[col] = df_final[col].fillna(0)

    return df_final

########################################################################
# post clean cleaning
########################################################################

def check_names(list_of_names, min_ratio = 90):

    print('\nchecking for typos\n')
    for pp in list_of_names:
        matches = fuzzywuzzy.process.extract(pp, list_of_names, limit=2,
                                scorer=fuzzywuzzy.fuzz.token_sort_ratio)
        if matches[1][1] > min_ratio:
            print(pp, ' - ', matches)

def replace_matches_in_column(df, column, string_to_match, min_ratio = 90):
    # get a list of unique strings
    strings = df[column].unique()

    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings,
                    limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches
    df.loc[rows_with_matches, column] = string_to_match

########################################################################
# Prep for machine learning
########################################################################

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
