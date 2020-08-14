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


def check_plant_names(df):

    plants = df.columns.tolist()
    for pp in plants:
        matches = fuzzywuzzy.process.extract(pp, plants, limit=2, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
        if matches[1][1] > 89:
            print(matches)

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

site_name = 'saltfl'
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
    #whole = clean.whole_to_ml(whole)

    species = xls.parse(spec_temp_string)
    species = clean.species_clean(species)
    cover, frequency = clean.get_abund_and_freq(species, column='DESC_LATIN')

    ground = xls.parse(ground_string)
    ground = clean.ground_clean(ground)

    year = whole.YEAR[0].astype(str)

    print(whole)

    df = pd.DataFrame()

    names = whole['BAP_BROAD'].unique()
    print(names)
    names = [x for x in names if str(x) != 'nan']
    for pp in names:
        print(pp)
        matches = fuzzywuzzy.process.extract(pp, names, limit=2, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
        if matches[1][1] > 89:
            print(matches)
    replace_matches_in_column(whole, 'BAP_BROAD', 'Supralittoral Sediment')

    df['year'] = whole['YEAR']
    df['BAP_B'] = whole['BAP_BROAD']
    df['BAP_P'] = whole['BAP_PRIORITY']
    df['freq_count'] = frequency.gt(0).sum(axis=1)
    data.append(df)

n_sites = len(data)


fig, axes = plt.subplots(nrows=n_sites)

for i, ax in zip(range(10), axes.flat):
    title = data[i]['year'][0]
    sns.boxplot(data=data[n_sites-i-1], x='freq_count', y='BAP_B', ax=ax).set_title(title)
plt.show()


fig, axes = plt.subplots(nrows=n_sites)

for i, ax in zip(range(10), axes.flat):
    title = data[i]['year'][0]
    sns.boxplot(data=data[n_sites-i-1], x='freq_count', y='BAP_P', ax=ax).set_title(title)
plt.show()


'''

xls = pd.ExcelFile(site_files[-1])

for name in xls.sheet_names:
    if 'whole' in name.lower():
        wpd_string = name
    if 'species te' in name.lower():
        spec_temp_string = name
    if 'ground' in name.lower():
        ground_string = name

whole = xls.parse(wpd_string)
whole = clean.whole_clean(whole)
#whole = clean.whole_to_ml(whole)

species = xls.parse(spec_temp_string)
species = clean.species_clean(species)
cover, frequency = clean.get_abund_and_freq(species, column='DESC_LATIN')

ground = xls.parse(ground_string)
ground = clean.ground_clean(ground)


df = pd.DataFrame()
df['BAP'] = whole['BAP_BROAD']
df['freq_count_13'] = frequency.gt(0).sum(axis=1)

print(df)

ax = sns.boxplot(data=df, x='freq_count', y='BAP')
plt.show()


# get the data from the other years and put all the counts into a new df
# get some of the data from whole for hue comparisons
# make a subplot
'''
