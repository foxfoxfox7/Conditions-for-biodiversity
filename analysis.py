import pickle
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import fuzzywuzzy
from fuzzywuzzy import process

import warnings
warnings.filterwarnings('ignore')

import clean


def check_names(list_of_names, min_ratio = 90):

    print('\nchecking for typos\n')
    for pp in list_of_names:
        matches = fuzzywuzzy.process.extract(pp, list_of_names, limit=2, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
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


with open("./data/file_list", "rb") as fp:
    b = pickle.load(fp)
surveys = [x for x in b if 'Metadata' not in x]

########################################################################
# Getting the data frames
########################################################################

site_name = 'lullington'
site_files = []
for file_name in surveys:
    if site_name.lower() in file_name.lower():
        site_files.append(file_name)
print('\n\n', site_files, '\n\n')

data = []
for site in site_files:
    xls = pd.ExcelFile(site)

    for name in xls.sheet_names:
        if 'whole' in name.lower():
            wpd_string = name
        if 'species te' in name.lower():
            spec_temp_string = name
        if 'ground' in name.lower():
            ground_string = name

    whole = xls.parse(wpd_string)
    whole = clean.whole_clean(whole)

    species = xls.parse(spec_temp_string)
    species = clean.species_clean(species)
    cover, frequency = clean.get_abund_and_freq(species, column='desc_latin')

    ground = xls.parse(ground_string)
    ground = clean.ground_clean(ground)

    year = whole['year'][0].astype(str)

    df = pd.DataFrame()

    ####################################################################
    # extracting the useful information
    ####################################################################

    df['year'] = whole['year']
    df['bap_b'] = whole['bap_broad']
    df['bap_p'] = whole['bap_priority']
    df['light'] = whole['light']
    df['wetness'] = whole['wetness']
    df['ph'] = whole['ph']
    df['fertility'] = whole['fertility']
    df['competitors'] = whole['competitors']
    df['stress'] = whole['stress']
    df['rudereals'] = whole['rudereals']
    df['nvc'] = whole['nvc_first']
    df['freq_count'] = frequency.gt(0).sum(axis=1)
    df['max_height'] = ground['max_height']
    df['median_height'] = ground['median_height']
    df['freq-bare soil'] = ground['freq-bare soil']
    data.append(df)

n_sites = len(data)
# Reverses the order so that it goes in chronological order
data2 = data[::-1]

########################################################################
# plotting the difference between habitats in box and whisker
########################################################################
'''
fig, axes = plt.subplots(nrows=n_sites)
fig.suptitle('species count')
for i, ax in zip(range(10), axes.flat):
    title = data2[i]['year'][0]
    sns.boxplot(data=data2[i], x='freq_count', y='bap_b', ax=ax).set_title(title)
plt.show()

fig, axes = plt.subplots(nrows=n_sites)
fig.suptitle('species count by bap bap_priority')
for i, ax in zip(range(10), axes.flat):
    title = data2[i]['year'][0]
    sns.boxplot(data=data2[i], x='freq_count', y='bap_p', ax=ax).set_title(title)
plt.show()

fig, axes = plt.subplots(nrows=n_sites)
fig.suptitle('sward heigh (max)')
for i, ax in zip(range(10), axes.flat):
    title = data2[i]['year'][0]
    sns.boxplot(data=data2[i], x='max_height', y='bap_b', ax=ax).set_title(title)
plt.show()

fig, axes = plt.subplots(nrows=n_sites)
fig.suptitle('sward heigh (median)')
for i, ax in zip(range(10), axes.flat):
    title = data2[i]['year'][0]
    sns.boxplot(data=data2[i], x='median_height', y='bap_b', ax=ax).set_title(title)
plt.show()

 Ask vic if combingin the BARE catagories would be good. so one for each section
fig, axes = plt.subplots(nrows=n_sites)
fig.suptitle('frequency of bare SOIL')
for i, ax in zip(range(10), axes.flat):
    title = data2[i]['year'][0]
    sns.boxplot(data=data2[i], x='freq-bare soil', y='bap_b', ax=ax).set_title(title)
plt.show()
'''
########################################################################
# concatting the data and fixing bap_b
########################################################################

total_data = pd.concat(data2)
print(total_data.head())

total_data = total_data[total_data['bap_b'].notna()]
total_data['bap_b'] = total_data['bap_b'].str.lower()

names = total_data['bap_b'].unique()
print('\nnames\n')
for nn in names:
    print(nn)
names = [x for x in names if str(x) != 'nan']
check_names(names, min_ratio = 80)

# Saltfleetby
#replace_matches_in_column(total_data, 'bap_b', 'fen, marsh and swamp', min_ratio = 80)
#replace_matches_in_column(total_data, 'bap_b', 'supralittoral sediment')

bap_b = total_data['bap_b'].unique()
fig_num = len(bap_b)

########################################################################
# plotting the difference between years in box and whisker
########################################################################

def plot_by_bap_vs_year(y_col, title = '', save = False, show = True):

    fig, axes = plt.subplots(ncols=fig_num, sharey=True, dpi=250)
    fig.suptitle(site_name + ' - ' + title)
    for i, ax in zip(range(10), axes.flat):
        sns.boxplot(data = total_data[total_data['bap_b'] == bap_b[i]],
            x='year', y=y_col, ax=ax).set_title(bap_b[i], fontsize=7)
    if save:
        plt.savefig('./figures/' + site_name + '_' + title + '.png')
    if show:
        plt.show()

#plot_by_bap_vs_year('freq_count', 'species_count')
#plot_by_bap_vs_year('max_height', 'sward_max')
#plot_by_bap_vs_year('median_height', 'sward med')
#plot_by_bap_vs_year('freq-bare soil', 'bare_soil_freq')
#plot_by_bap_vs_year('light', 'light')
#plot_by_bap_vs_year('wetness', 'wetness')
#plot_by_bap_vs_year('ph', 'ph')
#plot_by_bap_vs_year('fertility', 'fertility')
#plot_by_bap_vs_year('competitors', 'competitors')
#plot_by_bap_vs_year('stress', 'stress')
#plot_by_bap_vs_year('rudereals', 'rudereals')

########################################################################
# nvc
########################################################################

total_data['nvc'] = total_data['nvc'].str.partition(':')[0]
total_data['nvc'] = total_data['nvc'].str.replace('[a-z]', '')
total_data['nvc_edit'] = total_data['nvc'].str.replace('[0-9]', '')

#print(total_data['nvc'].value_counts())

years = total_data['year'].unique()
for yy in years:
    df = total_data[total_data['year'] == yy]
    print('\n', yy, '\n')
    print(df['nvc_edit'].value_counts())

nvc_types = total_data['nvc_edit'].value_counts()
nvc_types = nvc_types[nvc_types >=10]
nvc_t = nvc_types.index.tolist()

print(nvc_t)

def plot_by_nvc_vs_year(y_col, title = '', save = False, show = True):

    fig, axes = plt.subplots(ncols=fig_num, sharey=True, dpi=250)
    fig.suptitle(site_name + ' - ' + title)
    for i, ax in zip(range(10), axes.flat):
        sns.boxplot(data = total_data[total_data['nvc_edit'] == nvc_t[i]],
            x='year', y=y_col, ax=ax).set_title(nvc_t[i], fontsize=12)
    if save:
        plt.savefig('./figures/' + site_name + '_' + title + '.png')
    if show:
        plt.show()

#plot_by_nvc_vs_year('freq_count', 'species_count')
#plot_by_nvc_vs_year('max_height', 'sward_max')
#plot_by_nvc_vs_year('median_height', 'sward med')
#plot_by_nvc_vs_year('freq-bare soil', 'bare_soil_freq')
#plot_by_nvc_vs_year('light', 'light')
#plot_by_nvc_vs_year('wetness', 'wetness')
#plot_by_nvc_vs_year('ph', 'ph')
#plot_by_nvc_vs_year('fertility', 'fertility')
#plot_by_nvc_vs_year('competitors', 'competitors')
#plot_by_nvc_vs_year('stress', 'stress')
#plot_by_nvc_vs_year('rudereals', 'rudereals')
