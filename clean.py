import pickle
import re
import pandas as pd
import numpy as np


with open("./data/file_list", "rb") as fp:
    b = pickle.load(fp)
surveys = [x for x in b if 'Metadata' not in x]


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
            print(df.loc[i[0], 'PLOT_ID'])
            df.loc[i[0], 'PLOT_ID'] = re.sub("[^0-9]", "", df.loc[i[0], 'PLOT_ID'])
            print(df.loc[i[0], 'PLOT_ID'])

    df["PLOT_ID"] = df["PLOT_ID"].astype(float)

def whole_clean(df):

    df = df.dropna(how='all', axis = 1)
    df = df.dropna(how='all', axis = 0)

    remove_strings = ['note', 'comment', 'remark', 'date', 'data_issue',
        'qa', 'bng_grid', 'survey', 'nvc', 'strat']
    remove_cols = _get_list(df, remove_strings)
    df = df.drop(remove_cols, axis = 1)

    df["PLOT_ID"] = df["PLOT_ID"].astype(str)

    df = df.replace(r'^\s*$', np.NaN, regex=True)

    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    destring_strings = ['landuse', 'slopeform', 'scode', 'slope', 'aspect', 'altitude']
    destring_cols = _get_list(df, destring_strings)
    for col in destring_cols:
        for i in df[col].iteritems():
            if isinstance(i[1], str):
                print(df.loc[i[0], col], '!!!!!!!!!!!!!!!!!!!!!!!')
                if any(map(str.isdigit, i[1])):
                    df.loc[i[0], col] = re.search(r'\d+', i[1]).group()
                else:
                    df.loc[i[0], col] = np.NaN
                print(df.loc[i[0], col], '!!!!!!!!!!!!!!!!!!!!!!!')
        df[col] = df[col].astype(float)

    median_str = ['aspect', 'altitude', 'eastings', 'northings', 'slope']
    median_cols = _get_list(df, median_str, not_list=['slopeform'])
    for col in median_cols:
        df[col] = df[col].fillna(df[col].median())

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
    whole2 = pd.DataFrame({col:str(col)+'=' for col in df}, index=df.index) + df.astype(str)
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

    df = df.dropna(how='all', axis = 1)
    df = df.dropna(how='all', axis = 0)

    df["PLOT_ID"] = df["PLOT_ID"].astype(str)

    df = df.replace(r'^\s*$', np.NaN, regex=True)

    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    df = df[df['DESC_LATIN'] != 'Unidentified']

    remove_strings = ['data', 'comment', 'qa', 'cell', 'species_no']#'scode',
    remove_cols = _get_list(df, remove_strings)
    df = df.drop(remove_cols, axis = 1)

    destring_strings = ['percent', 'freq']
    destring_cols = _get_list(df, destring_strings)
    for col in destring_cols:
        for i in df[col].iteritems():
            if isinstance(i[1], str):
                print(df.loc[i[0], col], '!!!!!!!!!!!!!!!!!!!!!!!')
                if any(map(str.isdigit, i[1])):
                    df.loc[i[0], col] = re.search(r'\d+', i[1]).group()
                else:
                    df.loc[i[0], col] = np.NaN
                print(df.loc[i[0], col], '!!!!!!!!!!!!!!!!!!!!!!!')
        df[col] = df[col].astype(float)

    median_str = ['percent', 'frequency']
    median_cols = _get_list(df, median_str)
    for col in median_cols:
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna()

    return df

def get_abund_and_freq(df):

    df_dict = {num: df.loc[df['PLOT_ID'] == num] for num in df['PLOT_ID']}
    abundance_plots = []
    frequency_plots = []
    for key in df_dict.keys():
        df_dict[key] = df_dict[key].drop_duplicates(subset=['DESC_LATIN'], keep='first')

        abund_inst = pd.DataFrame()
        abund_inst = df_dict[key].pivot(index = 'PLOT_ID', columns = 'DESC_LATIN', values = 'PERCENT_COVER')
        abundance_plots.append(abund_inst)

        freq_inst = pd.DataFrame()
        freq_inst = df_dict[key].pivot(index = 'PLOT_ID', columns = 'DESC_LATIN', values = 'FREQUENCY')
        frequency_plots.append(freq_inst)

    df_abund = pd.concat(abundance_plots, axis=1)
    df_abund = df_abund.groupby(level=0, axis=1).sum()
    df_abund = df_abund.reset_index()#.rename(columns={df.index.name:'PLOT_ID'})
    df_freq = pd.concat(frequency_plots, axis=1)
    df_freq = df_freq.groupby(level=0, axis=1).sum()
    df_freq = df_freq.reset_index()#.rename(columns={df.index.name:'PLOT_ID'})

    return df_abund, df_freq


#site_name = 'braunton'
#site_files = []
#for file_name in surveys:
#    if site_name.lower() in file_name.lower():
#        site_files.append(file_name)
#print(site_files)
#print(site_files[-1])

#xls = pd.ExcelFile(site_files[-1])
#s_names = xls.sheet_names
#print(s_names)

#whole = xls.parse('Whole Plot Data')
#species = xls.parse('Species Template')
#ground = xls.parse('Ground Features')



for nn, ss in enumerate(surveys[10:20]):
    print('\n\n\n', nn, '\n\n\n', ss, '\n\n\n')
    xls = pd.ExcelFile(ss)
    #whole = xls.parse('Whole Plot Data')
    #whole = whole_clean(whole)
    #print(whole.info())
    #whole = whole_to_ml(whole)
    #print(whole.info())
    species = xls.parse('Species Template')
    species = species_clean(species)
    abundance, frequency = get_abund_and_freq(species)
    print(abundance.info())
    print(frequency.info())



#whole = whole_to_ml(whole)

#species = species_clean(species)
#abundance, frequency = get_abund_and_freq(species)

#print(whole.head())
#print(abundance.head())
#print(frequency.head())

