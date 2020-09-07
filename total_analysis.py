import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import itertools
import fuzzywuzzy
from fuzzywuzzy import process

import warnings
warnings.filterwarnings('ignore')

import clean


# reading in the dictionary with lists of habitats, speces etc
with open("./my_data/lookup.pkl", "rb") as fp:
    lookup = pickle.load(fp)

df = pd.read_pickle('./my_data/total_data.pkl')
print(df.head())
col_list = df.columns.tolist()

########################################################################
# cleaning bap broad
########################################################################

# getting the columns whihc are of object type (to format the strings)
obj_list = [col for col in col_list if df[col].dtype == object]
print('\nlist of object columns\n', obj_list)

# turning all strings to lower case due to discrepencies in input
for col in obj_list:
    df[col] = df[col].str.lower()

# looking at all the unique entries in bap_broad
baps = df['bap_broad'].unique()
baps = [x for x in baps if str(x) != 'nan']
print(len(baps), ' unique bap_broad habitats\n', sorted(baps))
# checking if any of them have names that are similat in case of typo
clean.check_names(baps, min_ratio = 80)

# from the dictionary, looking at the proper names for bap_broad
print(len(lookup['bap_broad']), ' lookup bap_broad\n', sorted(lookup['bap_broad']))

# These are the names which have typos which need to be replaced
clean.replace_matches_in_column(df, 'bap_broad', 'supralittoral sediment')
clean.replace_matches_in_column(df, 'bap_broad', 'fen, marsh and swamp', 80)
clean.replace_matches_in_column(df, 'bap_broad', 'dwarf shrub heath')
clean.replace_matches_in_column(df, 'bap_broad', 'bogs', 85)

# looking at the names afterward to double check for any more typos
baps = df['bap_broad'].unique()
baps = [x for x in baps if str(x) != 'nan']
print(len(baps), ' unique bap_broad habitats\n', sorted(baps))

# looking at how many times each of the unique bap_broad habitats occur
# many are not in the list so i think they are mistakenly assigned
# hopefully this means there are not many incidents of them
bap_types = df['bap_broad'].value_counts()
print(bap_types)
bap_types = bap_types[bap_types >=100]
bap_t = bap_types.index.tolist()

# leaving only the parst of the df with the important habitats
print('\nimportant habitats\n', bap_t)
df_bap = df[df['bap_broad'].isin(bap_t)]

########################################################################
# comparing values accross bap broad
########################################################################

# finding the columns with numerical entries to compare
float_list = [col for col in col_list if df[col].dtype == float]
print('\nlist of float columns\n', float_list)

# dropping a few columns that don't have any meaningful information
drop_l = ['freq-litter', 'median_height', 'max_height', 'year']
df_bap = df_bap.drop(drop_l, axis=1)
float_list = [val for val in float_list if val not in drop_l]
print(float_list)

#for var in float_list:
#    sns.lmplot(data=df_bap, x='freq_count', y=var, col='bap_broad', sharex=False)
#    plt.show()

########################################################################
# cleaning nvc
########################################################################

print(df.head())
print(df.info())

# removing the inofrmation on how good the nvc fit is
df['nvc_first'] = df['nvc_first'].str.partition(':')[0]
df['nvc_first'] = df['nvc_first'].str.partition('-')[0]

# the subdevisions are too specific to use. not enought samples
nums = ['0','1','2','3','4','5','6','7','8','9']
for nn in nums:
    df['nvc_first'] = df['nvc_first'].str.partition(nn)[0]

nvc_types = df['nvc_first'].value_counts()
nvc_types = nvc_types[nvc_types >=20]
nvc_t = nvc_types.index.tolist()

# leaving only the parst of the df with the important habitats
print('\nimportant nvc types\n', nvc_t)
df_nvc = df[df['nvc_first'].isin(nvc_t)]

########################################################################
# comparing values accross nvc types
########################################################################

# finding the columns with numerical entries to compare
float_list = [col for col in col_list if df[col].dtype == float]
print('\nlist of float columns\n', float_list)

# dropping a few columns that don't have any meaningful information
drop_l = ['freq-litter', 'median_height', 'max_height', 'year']
df_nvc = df_nvc.drop(drop_l, axis=1)
float_list = [val for val in float_list if val not in drop_l]
print(float_list)

for var in float_list:
    sns.lmplot(data=df_nvc, x='freq_count', y=var, col='nvc_first', sharex=False)#.set_title('lalala')
    plt.show()
