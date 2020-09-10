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

import sys
sys.path.append('..')

import clean


with open("../data/file_list", "rb") as fp:
    b = pickle.load(fp)
surveys = [x for x in b if 'Metadata' not in x]

with open("../my_data/lookup.pkl", "rb") as fp:
    lookup = pickle.load(fp)

########################################################################
# Getting the data frames
########################################################################

site_name = 'lulli'
site_files = []
for file_name in surveys:
    if site_name.lower() in file_name.lower():
        site_files.append('../' + file_name)
print('\n\n', site_files, '\n\n')

# Reverses the order so that it goes in chronological order
site_files = site_files[::-1]

data = []
spec_data = []
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

    df['freq_count'] = frequency.gt(0).sum(axis=1)

    extract_cols_w = ['year', 'bap_broad', 'bap_priority', 'light', 'wetness',
                      'ph', 'fertility', 'competitors', 'stress', 'rudereals',
                      'nvc_first']
    for col in extract_cols_w:
        try:
            df[col] = whole[col]
        except:
            df[col] = np.NaN

    extract_cols_g = ['max_height', 'median_height', 'freq-bare soil',
                      'freq-litter']
    for col in extract_cols_g:
        try:
            df[col] = ground[col]
        except:
            df[col] = np.NaN

    spec_data.append(cover)
    data.append(df)

n_sites = len(data)
df_t = pd.concat(data)
col_list = df_t.columns.tolist()


df_spec = pd.concat(spec_data)
df_spec.columns = df_spec.columns.str.lower()
spec_l = df_spec.columns.tolist()

########################################################################
# cleaning bap broad
########################################################################
'''
# getting the columns whihc are of object type (to format the strings)
obj_list = [col for col in col_list if df_t[col].dtype == object]
print('\nlist of object columns\n', obj_list)

# turning all strings to lower case due to discrepencies in input
for col in obj_list:
    df_t[col] = df_t[col].str.lower()

# looking at all the unique entries in bap_broad
baps = df_t['bap_broad'].unique()
baps = [x for x in baps if str(x) != 'nan']
print(len(baps), ' unique bap_broad habitats\n', sorted(baps))
# checking if any of them have names that are similat in case of typo
clean.check_names(baps, min_ratio = 80)

# from the dictionary, looking at the proper names for bap_broad
print(len(lookup['bap_broad']), ' lookup bap_broad\n', sorted(lookup['bap_broad']))

# These are the names which have typos which need to be replaced
clean.replace_matches_in_column(df_t, 'bap_broad', 'supralittoral sediment')
clean.replace_matches_in_column(df_t, 'bap_broad', 'fen, marsh and swamp', 80)
clean.replace_matches_in_column(df_t, 'bap_broad', 'dwarf shrub heath')
clean.replace_matches_in_column(df_t, 'bap_broad', 'bogs', 85)

# looking at the names afterward to double check for any more typos
baps = df_t['bap_broad'].unique()
baps = [x for x in baps if str(x) != 'nan']
print(len(baps), ' unique bap_broad habitats\n', sorted(baps))

# looking at how many times each of the unique bap_broad habitats occur
# many are not in the list so i think they are mistakenly assigned
# hopefully this means there are not many incidents of them
bap_types = df_t['bap_broad'].value_counts()
print(bap_types)
bap_types = bap_types[bap_types >=9]
bap_t = bap_types.index.tolist()

# leaving only the parst of the df with the important habitats
print('\nimportant habitats\n', bap_t)
df_bap = df_t[df_t['bap_broad'].isin(bap_t)]

# number of bap habitats of interest
bap_n = len(bap_t)
'''
########################################################################
# cleaning nvc
########################################################################

# removing the inofrmation on how good the nvc fit is
df_t['nvc_first'] = df_t['nvc_first'].str.partition(':')[0]
df_t['nvc_first'] = df_t['nvc_first'].str.partition('-')[0]

# the subdivisions are too specific to use. not enough samples
nums = ['0','1','2','3','4','5','6','7','8','9']
for nn in nums:
    df_t['nvc_first'] = df_t['nvc_first'].str.partition(nn)[0]

df_t['nvc_first'] = df_t['nvc_first'].replace({
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

# counting how many of each NVC types has been assigned
# only taking the types that have enough samples
nvc_types = df_t['nvc_first'].value_counts()
print(nvc_types)
nvc_types = nvc_types[nvc_types >=9]
nvc_t = nvc_types.index.tolist()

# leaving only the parts of the df with the important habitats
print('\nimportant nvc types\n', nvc_t)
df_nvc = df_t[df_t['nvc_first'].isin(nvc_t)]

# plotting the values counts of the habitats per year as some years
# may not have neough to draw valuable conclusions
years = df_nvc['year'].unique()
for yy in years:
    df_y = df_nvc[df_nvc['year'] == yy]
    print('\n', yy, '\n')
    print(df_y['nvc_first'].value_counts())

# number of nvc habitats of interest
nvc_n = len(nvc_t)

########################################################################
# plotting the difference between years in box and whisker
########################################################################
'''
def plot_by_bap_vs_year(y_col, title = '', save = False, show = True):

    fig, axes = plt.subplots(ncols=bap_n, sharey=True, dpi=250)
    fig.suptitle(site_name + ' - ' + title)
    for i, ax in zip(range(10), axes.flat):
        sns.boxplot(data = df_bap[df_bap['bap_broad'] == bap_t[i]],
            x='year', y=y_col, ax=ax).set_title(bap_t[i], fontsize=10)

    # have to go through each axes individually to set the angle
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=45)

    # turns the y axis off for all plots apart from first to save space
    for nn in range(bap_n-1):
        axes[nn+1].get_yaxis().set_visible(False)

    if save:
        plt.savefig('./figures/' + site_name + '_' + title + '.png')
    if show:
        plt.show()

plot_by_bap_vs_year('freq_count', 'species_count')
plot_by_bap_vs_year('max_height', 'sward_max')
plot_by_bap_vs_year('median_height', 'sward med')
plot_by_bap_vs_year('freq-bare soil', 'bare_soil_freq')
plot_by_bap_vs_year('light', 'light')
plot_by_bap_vs_year('wetness', 'wetness')
plot_by_bap_vs_year('ph', 'ph')
plot_by_bap_vs_year('fertility', 'fertility')
plot_by_bap_vs_year('competitors', 'competitors')
plot_by_bap_vs_year('stress', 'stress')
plot_by_bap_vs_year('rudereals', 'rudereals')

def plot_by_nvc_vs_year(y_col, title = '', save = False, show = True):

    fig, axes = plt.subplots(ncols=nvc_n, sharey=True, dpi=250)#
    fig.suptitle(site_name + ' - ' + title)
    for i, ax in zip(range(10), axes.flat):
        sns.boxplot(data = df_nvc[df_nvc['nvc_first'] == nvc_t[i]],
            x='year', y=y_col, ax=ax).set_title(nvc_t[i], fontsize=6)

    # have to go through each axes individually to set the angle
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=45)

    # turns the y axis off for all plots apart from first to save space
    for nn in range(nvc_n-1):
        axes[nn+1].get_yaxis().set_visible(False)

    if save:
        plt.savefig('./figures/' + site_name + '_' + title + '.png')
    if show:
        plt.show()


plot_by_nvc_vs_year('freq_count', 'species_count')
plot_by_nvc_vs_year('max_height', 'sward_max')
plot_by_nvc_vs_year('median_height', 'sward med')
plot_by_nvc_vs_year('freq-bare soil', 'bare_soil_freq')
plot_by_nvc_vs_year('light', 'light')
plot_by_nvc_vs_year('wetness', 'wetness')
plot_by_nvc_vs_year('ph', 'ph')
plot_by_nvc_vs_year('fertility', 'fertility')
plot_by_nvc_vs_year('competitors', 'competitors')
plot_by_nvc_vs_year('stress', 'stress')
plot_by_nvc_vs_year('rudereals', 'rudereals')
plot_by_nvc_vs_year('freq-litter', 'freq-litter')
'''
########################################################################
# looking for correlations across different habitats
########################################################################
'''
# finding the columns with numerical entries to compare
float_list = [col for col in col_list if df[col].dtype == float]
print('\nlist of float columns\n', float_list)

# dropping a few columns that don't have any meaningful information
drop_l = ['freq-litter', 'median_height', 'max_height', 'year']
df_bap = df_bap.drop(drop_l, axis=1)
float_list = [val for val in float_list if val not in drop_l]
print(float_list)

for var in float_list:
    sns.lmplot(data=df_bap, x='freq_count', y=var, col='bap_broad',
        sharex=False)
    plt.show()

# finding the columns with numerical entries to compare
float_list = [col for col in col_list if df[col].dtype == float]
print('\nlist of float columns\n', float_list)

# dropping a few columns that don't have any meaningful information
drop_l = ['freq-litter', 'median_height', 'max_height', 'year',
    'freq-bare soil']
df_nvc = df_nvc.drop(drop_l, axis=1)
float_list = [val for val in float_list if val not in drop_l]
print(float_list)

for var in float_list:
    sns.lmplot(data=df_nvc, x='freq_count', y=var, col='nvc_first',
        sharex=False)#.set_title('lalala')
    plt.show()
'''
########################################################################
# index = plot, columns = indicator list,
# values = [c] coverage in each plot, [p] presence in each plot
# normalised so that each plot coverage = 100%
########################################################################

with open("./indicators/indicator_d.pkl", "rb") as fp:
    indi = pickle.load(fp)

print('\nindicator species lists\n', indi.keys())

# the species that aren't in each survey will be NaNs
df_spec = df_spec.fillna(0.0)

# creating an empty dataframe to fill witht he indicator values
ind_vals = pd.DataFrame(index = df_spec.index)

# go through the indicator lists, counting the cover and presence of
# each species
for key, value in indi.items():
    ind_vals[key+'_[c]'] = df_spec[indi[key]].sum(axis=1)
    ind_vals[key+'_[p]'] = df_spec[indi[key]].gt(0).sum(axis=1)

########################################################################
# indicator species analysis
########################################################################

# picking which indicator lists to investigate
use_cols = ['(cg)']
useless_cols = ['[p]', 'neg10', 'gram', '2_pos']

cg_indicators = clean._get_list(ind_vals, use_cols, not_list = useless_cols)
ind_n = len(cg_indicators)

# df_ia has the same data as ind_vals but only the indicator lists
# of interest
df_ia = pd.DataFrame()
for ind in cg_indicators:
    df_ia[ind] = ind_vals[ind]

# including the year and nvc columns so that the plots can be diveded
df_ia['year'] = df_t['year']
df_ia['nvc'] = df_t['nvc_first']

# eliminating not chalk grassland plots
df_cg = df_ia[df_ia['nvc'] == 'CG']

# getting the years of the surveys and the number of plots
# for each year. the number of plots has to be reversed
years = df_cg['year'].unique().tolist()
year_nums = df_cg['year'].value_counts().tolist()
year_nums = year_nums[::-1]

# making a new dataframe whre the coumns are the years
# index is the chosen indicator lists
# values are the sum of all the plots for each year
df_ps = pd.DataFrame()
df_cs = pd.DataFrame()
for ii, yy in enumerate(years):
    df_ps[yy] = df_cg[df_cg['year'] == yy].astype(bool).sum()
    df_cs[yy] = df_cg[df_cg['year'] == yy].sum()

    # removing the year and nvc columns as they error with transformatio
    try:
        df_ps = df_ps.drop(['year', 'nvc'], axis = 0)
        df_cs = df_cs.drop(['year', 'nvc'], axis = 0)
    except:
        pass

    # normalising according to the number of sites
    df_ps[str(yy)+'_n'] = (df_ps[yy] / year_nums[ii]) * 100
    df_cs[str(yy)+'_n'] = df_cs[yy] / year_nums[ii]

print('\nnumber of sites with indicator species in + percentage\n')
print(df_ps)
print('\ncover of sites with indicator species, normalised by number of plots\n')
print(df_cs)

#fig, ax = plt.subplots(ncols=ind_n, figsize=(20, 6), sharey=False)
#for ii, ind in enumerate(cg_indicators):
#    sns.boxplot(data = df_cg, x='year', y=ind, ax=ax[ii])
#plt.show()

########################################################################
# indicator species analysis
########################################################################

# the dataframe to put only the species form an ind list in
ind_spec = pd.DataFrame()

# going through all the species in an indicator list only keeping them
for sp in indi['(cg)3_pos']:
    try:
        ind_spec[sp] = df_spec[sp]
    except:
        pass

# including the year and nvc columns so that the plots can be split
ind_spec['year'] = df_t['year']
ind_spec['nvc'] = df_t['nvc_first']

# eliminating not chalk grassland plots
ind_spec = ind_spec[ind_spec['nvc'] == 'CG']

# getting the years of the surveys and the number of plots
# for each year. the number of plots has to be reversed
years = df_cg['year'].unique().tolist()
year_nums = df_cg['year'].value_counts().tolist()
year_nums = year_nums[::-1]

# making a new dataframe whre the columns are the years
# index is the species in an indicator list
# values are the sum of each species across all the plots for each year
df_ps = pd.DataFrame()
df_cs = pd.DataFrame()
for yy in years:
    df_ps[yy] = ind_spec[ind_spec['year'] == yy].astype(bool).sum()
    df_cs[yy] = ind_spec[ind_spec['year'] == yy].sum()

    # removing the year and nvc columns as they error with transformatio
    try:
        df_ps = df_ps.drop(['year', 'nvc'], axis = 0)
        df_cs = df_cs.drop(['year', 'nvc'], axis = 0)
    except:
        pass

    # normalising according to the number of sites
    df_ps[str(yy)+'_n'] = (df_ps[yy] / year_nums[ii]) * 100
    df_cs[str(yy)+'_n'] = df_cs[yy] / year_nums[ii]

df_ps.loc['Total']= df_ps.sum()
df_cs.loc['Total']= df_cs.sum()

print('\nnumber of sites each indicator speces is in and percentage\n')
print(df_ps)
print('\ncover of the sites by each species, and accross all plots\n')
print(df_cs)

imp_sp = pd.DataFrame()
for yy in years:
    imp_sp[yy] = df_cs[str(yy)+'_n']
imp_sp = imp_sp.drop('Total', axis=0)

imp_sp = imp_sp.loc[(imp_sp > 1.5).any(axis=1)]
imp_sp = imp_sp.transpose()
imp_sp['Species'] = imp_sp.index


print(imp_sp)

df = imp_sp.melt('Species', var_name='% cover',  value_name='Year')
g = sns.factorplot(x="Species", y='Year', hue='% cover', data=df)
plt.show()
