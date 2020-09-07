import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import fuzzywuzzy
from fuzzywuzzy import process

import warnings
warnings.filterwarnings('ignore')

import clean


with open("./my_data/lookup.pkl", "rb") as fp:
    lookup = pickle.load(fp)

df = pd.read_pickle('./my_data/total_data.pkl')
print(df.head())
col_list = df.columns.tolist()

obj_list = [col for col in col_list if df[col].dtype == object]
print('\nlist of object columns\n', obj_list)
float_list = [col for col in col_list if df[col].dtype == float]
print('\nlist of float columns\n', float_list)

for col in obj_list:
    df[col] = df[col].str.lower()


baps = df['bap_broad'].unique()
baps = [x for x in baps if str(x) != 'nan']
baps2 = sorted(baps)
print('\nunique bap_broad habitats\n', baps2)
print(len(baps))
clean.check_names(baps, min_ratio = 80)

total_baps = sorted(lookup['bap_broad'])
print('\nlookup bap_broad\n', total_baps)
print(len(lookup['bap_broad']))

clean.replace_matches_in_column(df, 'bap_broad', 'supralittoral sediment')
clean.replace_matches_in_column(df, 'bap_broad', 'fen, marsh and swamp', 80)
clean.replace_matches_in_column(df, 'bap_broad', 'dwarf shrub heath')
clean.replace_matches_in_column(df, 'bap_broad', 'bogs', 85)

baps = df['bap_broad'].unique()
baps = [x for x in baps if str(x) != 'nan']
baps2 = sorted(baps)
print('\nunique bap_broad habitats\n', baps2)
print(len(baps))
clean.check_names(baps, min_ratio = 80)


'''
drop_l = ['freq-litter', 'median_height', 'max_height', 'year']
df = df.drop(drop_l, axis=1)
float_list = [val for val in float_list if val not in drop_l]
print(float_list)

lull_habs = ['calcareous grassland' 'dwarf shrub heath' 'scrub'
             'broadleaved, mixed and yew woodland']
lull_nvc = ['CG', 'W', 'MG', 'H', 'MC']

for var in float_list:
    sns.lmplot(data=df[df['bap_broad'].isin(lull_habs)], x='freq_count', y=var)
    plt.show()
'''
