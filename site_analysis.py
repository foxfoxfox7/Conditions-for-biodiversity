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


with open("./data/file_list", "rb") as fp:
    b = pickle.load(fp)
surveys = [x for x in b if 'Metadata' not in x]

with open("./my_data/lookup.pkl", "rb") as fp:
    lookup = pickle.load(fp)

########################################################################
# Getting the data frames
########################################################################

site_name = 'lullington'
site_files = []
for file_name in surveys:
    if site_name.lower() in file_name.lower():
        site_files.append(file_name)
print('\n\n', site_files, '\n\n')

# Reverses the order so that it goes in chronological order
site_files = site_files[::-1]

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

    print(whole.head())
    print(ground.head())
    #exit()

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
    df['freq-litter'] = ground['freq-litter']
    data.append(df)

n_sites = len(data)

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

total_data = pd.concat(data)

total_data = total_data[total_data['bap_b'].notna()]
total_data['bap_b'] = total_data['bap_b'].str.lower()

names = total_data['bap_b'].unique()
print('\nnames\n')
for nn in names:
    print(nn)
names = [x for x in names if str(x) != 'nan']
clean.check_names(names, min_ratio = 80)

# Saltfleetby
#clean.replace_matches_in_column(total_data, 'bap_b', 'fen, marsh and swamp', min_ratio = 80)
#clean.replace_matches_in_column(total_data, 'bap_b', 'supralittoral sediment')

bap_b = total_data['bap_b'].unique()
print(bap_b)
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
    #print('\n', yy, '\n')
    #print(df['nvc_edit'].value_counts())

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
#plot_by_nvc_vs_year('freq-litter', 'freq-litter')

########################################################################
# correlation
########################################################################


#g = sns.PairGrid(total_data, hue="year")
#g.map_diag(plt.hist)
#g.map_offdiag(plt.scatter)
#g.add_legend()
#plt.show()

col_list = total_data.columns.tolist()
col_list2 = [col for col in col_list if total_data[col].dtype == float]

#for var in col_list2:
#    sns.relplot(data=total_data, x='freq_count', y=var, hue='bap_b')
#    plt.show()

nvc_df = total_data[total_data['nvc_edit'].isin(nvc_t)]
drop_l = ['freq-litter', 'median_height', 'max_height']
nvc_df = nvc_df.drop(drop_l, axis=1)
col_list2 = [val for val in col_list2 if val not in drop_l]

for var in col_list2:
    sns.lmplot(data=nvc_df, x='freq_count', y=var, col='nvc_edit')
    plt.show()
