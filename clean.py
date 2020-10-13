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

    df_columns = df.columns.tolist()
    choose_col = []
    for col in df_columns:
        for r_str in list_names:
            if r_str.lower() in col.lower() :
                choose_col.append(col)

    # goes through each of the strings in not_list and looks for them in
    # the choose_cols
    for nl in not_list:
        choose_col = [ii for ii in choose_cols if nl.lower() not in ii.lower()]

    return choose_col

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
    # Make all the column titles lower case as there is variation amonst
    # sites
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

    # strips leading and trailing spaces from any columns that have
    # strings
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
    # Make all the column titles lower case as there is variation amonst
    # sites
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

    remove_strings = ['data', 'comment', 'qa', 'cell', 'species_no']
    remove_cols = _get_list(df, remove_strings)
    df = df.drop(remove_cols, axis = 1)

    # there are some strings in the percent cover which need to removed
    for i in df['percent_cover'].iteritems():
        if isinstance(i[1], str):
            df.loc[i[0], 'percent_cover'] = np.NaN
    df['percent_cover'] = df['percent_cover'].astype(float)

    # strips leading, trailing spaces from any columns that have strings
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

    # Make all col titles lower case as there is variation amonst sites
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
    # split the DF into one for each plot, transforming and combining
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

    # normalises so that each quadrat cover adds up to 100%
    #df_abund = df_abund.div((df_abund.sum(axis=1)/1), axis=0)

    return df_abund, df_freq

def ground_clean(df):
    '''
    Cleans the 'ground features' sheet in each survey excel file
    '''

    df = df[df['PLOT_ID'].notna()]
    df = df.dropna(how='all', axis = 1)
    df = df.dropna(how='all', axis = 0)
    df = df.rename(columns=lambda x: x.strip())
    # Make all col titles lower case as there is variation amonst sites
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

    # Creating a index id with year, site and plot so every single plot
    # has its own index
    year_str = str(int(df['year'][0]))
    df['index_id'] = df['plot_id'] + '_' + df['sitecode'] + '_' + year_str

    # Some of the plots dont have frequency and only have percent cover,
    # Changed it to be 9001 like below, so it sticks till the end of
    # data wrangling and can be converted back to NaNs
    # change to string it gets lost when summing in get_abund_and_freq
    # changing it to NaN it gets converted to 0 later
    if 'frequency' not in df:
        df['frequency'] = 9001 #df['percent_cover'] / 4
        print('\n\nplot ' + str(df['sitecode'][0]) + str(df['year'][0])
         + ' with no frequency')

    # replace blank cells with 9001 as they will need to be converted to
    # nans after everything else is done. only a number will stick
    zero_cols = _get_list(df, ['freq', 'cover'])
    for col in zero_cols:
        df[col] = df[col].fillna(9001)

    # taking the trailing and leading spaces from the features
    df['feature'] = df['feature'].apply(lambda x: x.strip())
    # change all feature labels to lower case to cover for discrepancies
    df['feature'] = df['feature'].str.lower()

    # there is 1 veg height for each plot so is taken as the DF base
    df_height = df[df['feature'].str.lower() == 'vegetation height']
    # The other catagories need to be transformed into a pivot table
    df_other = df[df['feature'].str.lower() != 'vegetation height']
    # Function which generates pivot tables
    # Used if there are multiple entries for each plot.1 plot - 1 row
    g_cover, g_frequency = get_abund_and_freq(df_other, column='feature')

    # Corrects for the files that don't have any height data. we still
    # need the index if for merging
    df_height['index_id'] = df['index_id']

    # Some of the surveys have many cells replaced with '*'. This ruins
    # the median and max calculations later on
    zero_cols = _get_list(df_height, ['cell'])
    for col in zero_cols:
        try:
            df_height[col] = df_height[col].replace({"*": np.NaN})
        except:
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

    # some plots that have multiple entries This drops al but the 1st
    df_height = df_height[df_height.duplicated('index_id',
                                                        keep='first') == False]

    # remove useless columns. done late as 'cell' is needed before
    remove_strings = ['percent', 'freq', 'data', 'cell', 'feature', 'site',
        'code', 'year', 'qa', 'plot_id']
    remove_cols = _get_list(df_height, remove_strings)
    df_height = df_height.drop(remove_cols, axis = 1)

    # keeping the essential information to add our transformed DFs to
    df_base = pd.DataFrame()
    base_strings = ['sitec', 'mcode', 'index', 'plot_id']
    base_cols = _get_list(df, base_strings)
    for col in base_cols:
        df_base[col] = df[col]
    df_base = df_base[df_base.duplicated('index_id', keep='first') == False]

    # Setting the same index before combining the DFs
    # index for cover and frequency are already set during pivot table
    df_base = df_base.set_index('index_id')
    df_height = df_height.set_index('index_id')

    # labelling col titles separately for when they need to be combined
    g_cover = g_cover.add_prefix('cover-')
    g_frequency = g_frequency.add_prefix('freq-')

    df_final = pd.concat((df_base, df_height, g_frequency, g_cover), axis = 1)

    # These are columns where the row in the data is empty so it gives
    # nan but it should be 0
    zero_cols = _get_list(df_final, ['freq', 'cover'])
    for col in zero_cols:
        df_final[col] = df_final[col].fillna(0)

    # REplace the 9001s with nans as these shoudl be nans they are from
    # cells which had no number but which should have one
    # on certain surveys these can be replaced, on others they can't
    for col in zero_cols:
        a = df_final[col].values
        df_final[col] = np.where(a > 9000, np.nan, a).tolist()

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
    close_matches =\
         [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches
    df.loc[rows_with_matches, column] = string_to_match

def clean_bap_broad(df, lookup):
    '''
    Cleans the bap_broad column whihc is a useful way to group the plots
    There are lots of typos in this column as it is all made of
    surveyor input strings. Must also pass the dictionary with bap list
    '''

    col_list = df.columns.tolist()

    # getting the cols whihc are of object type (to format the strings)
    obj_list = [col for col in col_list if df[col].dtype == object]
    print('\nlist of object columns\n', obj_list)

    # turning all strings to lower case due to discrepencies in input
    for col in obj_list:
        df[col] = df[col].str.lower()

    # looking at all the unique entries in bap_broad
    baps = df['bap_broad'].unique()
    baps = [x for x in baps if str(x) != 'nan']
    print(len(baps), ' unique bap_broad habitats\n', sorted(baps))
    # check if any of them have names that are similat in case of typo
    check_names(baps, min_ratio = 80)

    # from the dictionary, looking at the proper names for bap_broad
    print(len(lookup['bap_broad']),
     ' lookup bap_broad\n', sorted(lookup['bap_broad']))

    # These are the names which have typos which need to be replaced
    replace_matches_in_column(df, 'bap_broad', 'supralittoral sediment')
    replace_matches_in_column(df, 'bap_broad', 'fen, marsh and swamp', 80)
    replace_matches_in_column(df, 'bap_broad', 'dwarf shrub heath')
    replace_matches_in_column(df, 'bap_broad', 'bogs', 85)

    # looking at the names afterward to double check for any more typos
    baps = df['bap_broad'].unique()
    baps = [x for x in baps if str(x) != 'nan']
    print(len(baps), ' unique bap_broad habitats\n', sorted(baps))

    return df

def clean_nvc(df):
    '''
    nvc column has lots of data in it. this function removes the
    useless stuff and separates the useful stuff
    '''

    # removing the inofrmation on how good the nvc fit is
    df['nvc_first'] = df['nvc_first'].str.partition(':')[0]
    df['nvc_first'] = df['nvc_first'].str.partition('-')[0]
    #df['nvc_tot'] = df['nvc_first']

    # the subdivisions are too specific to use. not enough samples
    for i in df['nvc_first'].iteritems():
        df.loc[i[0], 'nvc_first'] = re.sub(r'\D+$', '', i[1])

    # Then split into the nums and letters seperately for nvc analysis
    df['nvc_num'] = df['nvc_first']
    for i in df['nvc_num'].iteritems():
        df.loc[i[0], 'nvc_num'] = re.sub(r'(^[^\d]+)', '', i[1])

    df['nvc_let'] = df['nvc_first']
    for i in df['nvc_let'].iteritems():
        df.loc[i[0], 'nvc_let'] = re.sub(r"[^a-zA-Z]", '', i[1])

    df['nvc_let'] = df['nvc_let'].str.lower()
    df['nvc_let'] = df['nvc_let'].replace({
        "w": 'woodlands and scrub',
        'm': 'mires',
        "h": 'heathes',
        'mg': 'mesotrophic grasslands',
        'cg': 'calcicolous grasslands',
        'u': 'calcifugous grasslands',
        'a': 'aquatic communities',
        's': 'swamps and tall herb ferns',
        'sd': 'shingle, sandline and sand-dune',
        'sm': 'salt marsh',
        'mc': 'maritime cliff',
        'ov': 'vegetation of open habitats'
        })

    #df['nvc_let'] = df['nvc_let'].str.capitalize()

    return df


########################################################################
# Prep for machine learning
########################################################################

def whole_to_ml(df):
    '''
    this function for turning the whole plot data sheet into a df
    which can be used for ml was made right at the begninng.
    as such it needs updating
    '''

    # Removes a few of the oclumsn taht arent useful for ml
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
        # Some of the sheets have certain code columns and some dont
        try:
            site_dummies = pd.concat(dummies, axis = 1)
            df = df.drop(dummies_cols, axis = 1)
            site_dummies = site_dummies.groupby(level=0, axis=1).sum()
            site_dummies = site_dummies.add_prefix(code_str + '=')
            dummy_dfs.append(site_dummies)
        except ValueError:
            print('no scode columns')

    # giving the new columns instructive names
    restring_strings = ['code']
    restring_cols = _get_list(df, restring_strings)
    whole2 = pd.DataFrame({col:str(col)+'=' for col in df},
     index=df.index) + df.astype(str)
    df = df.drop(restring_cols, axis = 1)
    for col in restring_cols:
        df[col] = whole2[col]

    # These may have useful info so we create dummy cols (0 or 1)
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
