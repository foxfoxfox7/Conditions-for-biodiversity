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

    extract_cols_g = ['max_height', 'median_height', 'cover-bare soil',
                      'cover-litter']
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

print(df_t['year'].value_counts())

########################################################################
# cleaning bap broad
########################################################################

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

# showing the values counts of the habitats per year as some years
# may not have neough to draw valuable conclusions
years = df_bap['year'].unique()
for yy in years:
    df_y = df_bap[df_bap['year'] == yy]
    print('\n', yy, '\n')
    print(df_y['bap_broad'].value_counts())

# number of bap habitats of interest
bap_n = len(bap_t)

########################################################################
# cleaning nvc
########################################################################

# removing the inofrmation on how good the nvc fit is
df_t['nvc_first'] = df_t['nvc_first'].str.partition(':')[0]
df_t['nvc_first'] = df_t['nvc_first'].str.partition('-')[0]
#df_t['nvc_tot'] = df_t['nvc_first']


# the subdivisions are too specific to use. not enough samples
for i in df_t['nvc_first'].iteritems():
    df_t.loc[i[0], 'nvc_first'] = re.sub(r'\D+$', '', i[1])

# Then split into jus the numbers and letters seperately for nvc analasi
df_t['nvc_num'] = df_t['nvc_first']
for i in df_t['nvc_num'].iteritems():
    df_t.loc[i[0], 'nvc_num'] = re.sub(r'(^[^\d]+)', '', i[1])

df_t['nvc_let'] = df_t['nvc_first']
for i in df_t['nvc_let'].iteritems():
    df_t.loc[i[0], 'nvc_let'] = re.sub(r"[^a-zA-Z]", '', i[1])

df_t['nvc_let'] = df_t['nvc_let'].str.lower()
df_t['nvc_let'] = df_t['nvc_let'].replace({
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
print(df_t.head())
# counting how many of each NVC types has been assigned
# only taking the types that have enough samples

#df_t['nvc_let'] = df_t['nvc_let'].str.capitalize()

nvc_types = df_t['nvc_let'].value_counts()
print(nvc_types)
nvc_types = nvc_types[nvc_types >=9]
nvc_t = nvc_types.index.tolist()

# leaving only the parts of the df with the important habitats
print('\nimportant nvc types\n', nvc_t)
df_nvc = df_t[df_t['nvc_let'].isin(nvc_t)]

# showing the values counts of the habitats per year as some years
# may not have neough to draw valuable conclusions
years = df_nvc['year'].unique()
for yy in years:
    df_y = df_nvc[df_nvc['year'] == yy]
    print('\n', yy, '\n')
    print(df_y['nvc_let'].value_counts())

# number of nvc habitats of interest
nvc_n = len(nvc_t)

########################################################################
# plotting the difference between years in box and whisker
########################################################################

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

#plot_by_bap_vs_year('freq_count', 'species_count')
#plot_by_bap_vs_year('max_height', 'sward_max')
#plot_by_bap_vs_year('median_height', 'sward med')
#plot_by_bap_vs_year('cover-bare soil', 'bare_soil_freq')
#plot_by_bap_vs_year('light', 'light')
#plot_by_bap_vs_year('wetness', 'wetness')
#plot_by_bap_vs_year('ph', 'ph')
#plot_by_bap_vs_year('fertility', 'fertility')
#plot_by_bap_vs_year('competitors', 'competitors')
#plot_by_bap_vs_year('stress', 'stress')
#plot_by_bap_vs_year('rudereals', 'rudereals')

def plot_by_nvc_vs_year(
        y_col, title = '', save = False, show = True, fig_name = ''
        ):

    fig, axes = plt.subplots(ncols=nvc_n, sharey=True, figsize=(10, 4), dpi=240)#
    fig.suptitle(title, fontsize=14)
    for i, ax in zip(range(10), axes.flat):
        sns.boxplot(data = df_nvc[df_nvc['nvc_let'] == nvc_t[i]],
            x='year', y=y_col, ax=ax).set_title(nvc_t[i], fontsize=9)



    # have to go through each axes individually to set the angle
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=45)
        ax.set(ylim=(0, 11))

    # turns the y axis off for all plots apart from first to save space
    for nn in range(nvc_n-1):
        axes[nn+1].get_yaxis().set_visible(False)

    if save:
        plt.savefig('./figures/' + fig_name + '.png')
    if show:
        plt.show()

#df_nvc['Number of species'] = df_nvc['freq_count']
#plot_by_nvc_vs_year('Number of species', 'Species richness', save=True, fig_name='lull_spec_count')


#df_nvc['Height (cm)'] = df_nvc['median_height']
#df_nvc.loc[df_nvc['year'] == 2014, 'Height (cm)'] = np.NaN
#plot_by_nvc_vs_year('Height (cm)', 'Vegetation height', save = True, fig_name='heights')

#df_nvc['Bare soil cover (%)'] = df_nvc['cover-bare soil']
#plot_by_nvc_vs_year('Bare soil cover (%)', 'Bare soil', save=True, fig_name='bare_soil')

#df_nvc['Litter cover (%)'] = df_nvc['cover-litter']
#plot_by_nvc_vs_year('Litter cover (%)', 'Litter', save=True, fig_name='litter')

#plot_by_nvc_vs_year('max_height', 'sward_max')
#plot_by_nvc_vs_year('light', 'light')
#plot_by_nvc_vs_year('wetness', 'wetness')
#plot_by_nvc_vs_year('ph', 'ph')
#plot_by_nvc_vs_year('fertility', 'fertility')
#plot_by_nvc_vs_year('competitors', 'competitors')
#plot_by_nvc_vs_year('stress', 'stress')
#plot_by_nvc_vs_year('rudereals', 'rudereals')


########################################################################
# looking for correlations across different habitats
########################################################################
'''
# finding the columns with numerical entries to compare
float_list = [col for col in col_list if df[col].dtype == float]
print('\nlist of float columns\n', float_list)

# dropping a few columns that don't have any meaningful information
drop_l = ['cover-litter', 'median_height', 'max_height', 'year']
df_bap = df_bap.drop(drop_l, axis=1)
float_list = [val for val in float_list if val not in drop_l]
print(float_list)

#for var in float_list:
#    sns.lmplot(data=df_bap, x='freq_count', y=var, col='bap_broad',
#        sharex=False)
#    plt.show()

# finding the columns with numerical entries to compare
float_list = [col for col in col_list if df[col].dtype == float]
print('\nlist of float columns\n', float_list)

# dropping a few columns that don't have any meaningful information
drop_l = ['cover-litter', 'median_height', 'max_height', 'year',
    'cover-bare soil']
df_nvc = df_nvc.drop(drop_l, axis=1)
float_list = [val for val in float_list if val not in drop_l]
print(float_list)

#for var in float_list:
#    sns.lmplot(data=df_nvc, x='freq_count', y=var, col='nvc_let',
#        sharex=False)#.set_title('lalala')
#    plt.show()
'''
########################################################################
# index = plot, columns = indicator list,
# values = [c] coverage in each plot, [p] presence in each plot
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
'''
# picking which indicator lists to investigate
use_cols = ['(cg)']#, '(cg) tree scrub']
useless_cols = ['[p]']#, 'exotic species', 'Acrocarpous mosses', 'bracken',
#   'herbaceous', 'ulex', 'dwarf']

cg_indicators = clean._get_list(ind_vals, use_cols, not_list = useless_cols)
ind_n = len(cg_indicators)

# df_ia has the same data as ind_vals but only the indicator lists
# of interest
df_ia = pd.DataFrame()
for ind in cg_indicators:
    df_ia[ind] = ind_vals[ind]

# Changing the first column name for the figure
#cg_indicators = ['Proportion of plant cover', '(h) graminoids_[c]', '(h) forbs_[c]', '(cg) tree scrub_[c]']
#df_ia.columns = cg_indicators
#print(df_ia.columns.tolist())

# including the year and nvc columns so that the plots can be diveded
df_ia['year'] = df_t['year']
df_ia['nvc'] = df_t['nvc_let']

# eliminating not chalk grassland plots
df_cg = df_ia[df_ia['nvc'] == 'calcicolous grasslands'] #calcicolous grasslands

# getting the years of the surveys and the number of plots
# for each year. makes a df and then tkaes the numbers in the nvc col
years = df_cg['year'].unique().tolist()
df_cg_y_n = df_cg.groupby('year').count()
year_nums = df_cg_y_n['nvc'].tolist()

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

H_titles = ['Bryohpytes and Lichens', 'Graminoids', 'Forbs', 'Tree and scrub']

#fig, ax = plt.subplots(ncols=ind_n, sharey=False, figsize=(14, 6), dpi = 140)
#fig.suptitle('Plant composition in calcicolous grasslands communities', fontsize=20)
#for ii, ind in enumerate(cg_indicators):
#    sns.boxplot(data = df_cg, x='year', y=ind, ax=ax[ii],
#        showfliers = False).set_title(H_titles[ii], fontsize=14)
#
#for ax in fig.axes:
#    plt.sca(ax)
    #plt.xticks(rotation=45)
    #ax.set(ylim=(0, 11))
#
#for nn in range(nvc_n-1):
#    fig.axes[nn+1].set(ylabel=None)#, fontsize=18)
#    #fig.axes[nn+1].get_yaxis().set_visible(False)
#plt.savefig('./figures/cg_plant_comp.png')
#plt.show()
'''
########################################################################
# indicator species analysis
########################################################################
'''
# the dataframe to put only the species form an ind list in
ind_spec = pd.DataFrame()

print(indi.keys())

# going through all the species in an indicator list only keeping them
for sp in indi['(cg)3_pos']:
    try:
        ind_spec[sp] = df_spec[sp]
    except:
        pass

# including the year and nvc columns so that the plots can be split
ind_spec['year'] = df_t['year']
ind_spec['nvc'] = df_t['nvc_let']

# eliminating not chalk grassland plots
df_cg = ind_spec[ind_spec['nvc'] == 'calcicolous grasslands']

# getting the years of the surveys and the number of plots
# for each year. the number of plots has to be reversed
years = df_cg['year'].unique().tolist()
df_cg_y_n = df_cg.groupby('year').count()
year_nums = df_cg_y_n['nvc'].tolist()

# making a new dataframe whre the columns are the years
# index is the species in an indicator list
# values are the sum of each species across all the plots for each year
df_ps = pd.DataFrame()
df_cs = pd.DataFrame()
for ii, yy in enumerate(years):
#for yy in years:
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

df_ps.loc['Total']= df_ps.sum()
df_cs.loc['Total']= df_cs.sum()

print('\nnumber of sites each indicator speces is in and percentage\n')
print(df_ps)
print('\ncover of the sites by each species, and accross all plots\n')
print(df_cs)

df_ps = df_ps.drop('Total', axis=0)
df_cs = df_cs.drop('Total', axis=0)

imp_sp = pd.DataFrame()
for yy in years:
    imp_sp[yy] = df_cs[str(yy)+'_n']

imp_sp = imp_sp.loc[(imp_sp > 0.015).any(axis=1)]
imp_sp = imp_sp.transpose()
imp_sp['Species'] = imp_sp.index

print(imp_sp)

df = imp_sp.melt('Species', var_name='% cover',  value_name='Year')

fig, ax = plt.subplots()
ax = sns.factorplot(x="Species", y='Year', hue='% cover', data=df)
plt.show()
'''
########################################################################
# nvc_type analysis
########################################################################
'''
print(nvc_t)
df_t['nvc_num'] = df_t['nvc_num'].replace({
    '6': 'CG6',
    '4': 'CG4',
    "3": 'CG3',
    '2': 'CG2'
#    '6': 'H6',
#    '7': 'H7',
#    '8': 'H8'
    })

def plot_NVC_makeup(df, nvc_type, save=False, fig_name='dummy', savename='d'):
    dfn = df[df['nvc_let'] == nvc_type]

    print('\nnumber of ', nvc_type, ' plots in each year\n')
    print(dfn['year'].value_counts())

    # separating the years, counting instances of each nvc
    df_nvc_l = []
    for yy in years:
        dfy = dfn[dfn['year'] == yy]
        val_c = dfy['nvc_num'].value_counts()
        df_nvc_l.append(val_c)

    # creating the total df for the nvc type, renaming columns to years
    df_nvc_a = pd.concat(df_nvc_l, axis = 1)
    df_nvc_a = df_nvc_a.fillna(0)
    df_nvc_a.columns = years

    print('\nnumber of ', nvc_type,  ' subtype type in each year\n')
    print(df_nvc_a)

    # normalizing across columns so its a percentage
    for yy in years:
        df_nvc_a[yy] = (df_nvc_a[yy] / df_nvc_a[yy].sum()) * 100

    fig, ax = plt.subplots(dpi=200, figsize=(10,4))
    df_nvc_a = df_nvc_a.transpose()
    df_nvc_a.plot(kind='bar', stacked=True, ax = ax).set_title(fig_name)
    plt.ylabel('Percentage of total')
    plt.xticks(rotation=0)

    if save:
        plt.savefig('./figures/' + savename + '.png')
    plt.show()

plot_NVC_makeup(df_t, 'calcicolous grasslands', save=True,
    fig_name='NVC sub-communities for Calcicolous grasslands',
    savename='CG_nvc_sub')

#plot_NVC_makeup(df_t, 'heathes', save=True,
#    fig_name='NVC sub-communities for Heathes',
#    savename='H_nvc_sub')

plot_NVC_makeup(df_t, 'mesotrophic grasslands', save=False,
    fig_name='NVC sub-communities for Mesotrophic grasslands',
    savename='CG_nvc_sub')

plot_NVC_makeup(df_t, 'woodlands and scrub', save=False,
    fig_name='NVC sub-communities for Woodlands and scrub',
    savename='CG_nvc_sub')
'''
########################################################################
# nvc species analysis
########################################################################
'''
#91–100% 10
#76–90% 9
#51–75% 8
#34–50% 7
#26–33% 6
#11–25% 5
#4–10% 4
#<4% (many individuals) 3
#<4% (several individuals) 2
#<4% (few individuals) 1

#I = 1–20% (i.e. 1 stand in 5) scarce
#II = 21–40% occasional
#III = 41–60% frequent
#IV = 61–80% constant
#V = 81–100% constant

with open("../my_data/nvc_spec.pkl", "rb") as fp:
    nvc_cat_spec = pickle.load(fp)

# To take from teh dataframe only a specific habitat
#not_cg_cols = clean._get_list(nvc_cat_spec, [''], ['CG'])
#cg_cols = clean._get_list(nvc_cat_spec, ['CG'])
#nvc_cat_spec = nvc_cat_spec.drop(not_cg_cols, axis=1)

#print(nvc_cat_spec.columns.tolist())

#test_list = clean._get_list(nvc_cat_spec, 'CG2')
#print(test_list)


def make_nvc_df(df, nvc_type):

    drop_cols = clean._get_list(df, [''], [nvc_type])
    df = df.drop(drop_cols, axis = 1)
    df = df.dropna()
    further_drop = clean._get_list(df, ['[c]'])
    df = df.drop(further_drop, axis = 1)

    # setting up the nvc list df so that it matches the plots species
    df[nvc_type] = df[nvc_type].str.strip()
    df[nvc_type] = df[nvc_type].str.lower()
    df = df.set_index(nvc_type)

    return df


df_n2 = make_nvc_df(nvc_cat_spec, 'CG2')

# including the nvc type to only take the type matching the selected nvc
print(df_t.head())
df_spec['nvc'] = df_t['nvc_let']
df_spec = df_spec[df_spec['nvc'] == 'calcicolous grasslands']
df_spec = df_spec.drop('nvc', axis = 1)


df_spec['year'] = df_t['year']


# rearanging the plots df so the index is the species
df_spec_trans = df_spec.transpose()

# finding the species that are in the plots df and the nvc df
in_both = [sp for sp in df_spec_trans.index.tolist() if sp in df_n2.index.tolist()]

# Arranging the indexes of nvc df and plots df so they are the same
df_spec_trans = df_spec_trans[df_spec_trans.index.isin(in_both)]
df_spec_trans = df_spec_trans.sort_index()
df_n2 = df_n2[df_n2.index.isin(in_both)]
df_n2 = df_n2.sort_index()


df_comb = pd.concat([df_n2, df_spec_trans], axis = 1)
print(df_comb)

#df_n2 = df_n2[df_n2.index.isin(all_cg_spec)]
#print(df_n2)
'''
########################################################################
# individual species
########################################################################

spec_ind = ['ulex', 'Genista']
spec_ind = ['achillea millefolium']
#spec_ind = indi['(h) forbs']
habitat = 'calcicolous grasslands'
#habitat = 'heath'

spec_cols = clean._get_list(df_spec, spec_ind)
print(spec_cols)

# the dataframe to put only the species form an ind list in
df_spec_ind = pd.DataFrame()

# going through all the species in an indicator list only keeping them
for sp in spec_cols:
    try:
        df_spec_ind[sp] = df_spec[sp]
    except:
        pass

# including the year and nvc columns so that the plots can be split
df_spec_ind['year'] = df_t['year']
df_spec_ind['nvc'] = df_t['nvc_let']

df_spec_ind = df_spec_ind[df_spec_ind['nvc'] == habitat]

# getting the years of the surveys and the number of plots
# for each year. the number of plots has to be reversed
years = df_spec_ind['year'].unique().tolist()
df_y_n = df_spec_ind.groupby('year').count()
year_nums = df_y_n['nvc'].tolist()

# making a new dataframe whre the columns are the years
# index is the species in an indicator list
# values are the sum of each species across all the plots for each year
df_ps = pd.DataFrame()
df_cs = pd.DataFrame()
for ii, yy in enumerate(years):
    df_ps[yy] = df_spec_ind[df_spec_ind['year'] == yy].astype(bool).sum()
    df_cs[yy] = df_spec_ind[df_spec_ind['year'] == yy].sum()

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

df_ps = df_ps.drop('Total', axis=0)
df_cs = df_cs.drop('Total', axis=0)

imp_sp = pd.DataFrame()
for yy in years:
    imp_sp[yy] = df_cs[str(yy)+'_n']

imp_sp = imp_sp.loc[(imp_sp > 0.0001).any(axis=1)]
imp_sp = imp_sp.transpose()
imp_sp['Species'] = imp_sp.index

print(imp_sp)


df = imp_sp.melt('Species', var_name='% cover',  value_name='Year')
g = sns.factorplot(x="Species", y='Year', hue='% cover', data=df)
plt.show()

