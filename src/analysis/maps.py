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

    whole = xls.parse(wpd_string)
    whole = clean.whole_clean(whole)

    df = pd.DataFrame()

    extract_cols_w = ['year', 'bap_broad', 'bap_priority', 'nvc_first',
        'eastings', 'northings']
    for col in extract_cols_w:
        try:
            df[col] = whole[col]
        except:
            df[col] = np.NaN

    data.append(df)

n_sites = len(data)
df_t = pd.concat(data)
col_list = df_t.columns.tolist()

# Cleans all typos in the bap broad column
df_t = clean.clean_bap_broad(df_t, lookup)

# separates the parts of the nvc column so they are usable
df_t = df_t[df_t['nvc_first'].notna()]
df_t = clean.clean_nvc(df_t)

print(df_t.head())

########################################################################
# obtaining the map data
########################################################################

# eastings and northings convered to a csv file to convert into
# longitude and latitude thorugh an external site
# https://gridreferencefinder.com/batchConvert/batchConvert.php
#coord_conv = pd.DataFrame()
#coord_conv['eastings'] = df_t['eastings']
#coord_conv['northings'] = df_t['northings']
#coord_conv['plot_id'] = df_t.index
#coord_conv.to_csv('coords_in.csv', header=False, index=False)

# reads in latitude and longitude from the csv file created externally
df_coords = pd.read_csv('coords_out.csv', index_col = 2, header=None)
df_t['longitude'] = df_coords[4]
df_t['latitude'] = df_coords[5]
print(df_t.head())

min_lat = df_t['latitude'].min()
max_lat = df_t['latitude'].max()
min_long = df_t['longitude'].min()
max_long = df_t['longitude'].max()

lat_extra = (max_lat - min_lat) / 10
long_extra = (max_long - min_long) / 10

map_lat_min = min_lat - lat_extra
map_lat_max = max_lat + lat_extra
map_long_min = min_long - long_extra
map_long_max = max_long + long_extra
box = [map_lat_min, map_lat_max, map_long_min, map_long_max]

# Use an external site to download a map of the area with these coords
# https://www.openstreetmap.org/
print('map coords - ', map_lat_min, map_lat_max, map_long_min, map_long_max)

# reads in the image of the map for use in figures
site_map = plt.imread('./map_lull2.png')

########################################################################
# obtaining the groupings for analysis
########################################################################

# The information for splitting up the plots on the figure
colours = ['r', 'y', 'c', 'b', 'c', 'm', 'y', 'k', 'w']
years = df_t['year'].unique().tolist()

# counting how many of each NVC types has been assigned
# only taking the types that have enough samples
nvc_types = df_t['nvc_let'].value_counts()
print(nvc_types)
nvc_types = nvc_types[nvc_types >=9]
nvc_t = nvc_types.index.tolist()
print('nvc communities - ', nvc_t)

bap_types = df_t['bap_broad'].value_counts()
print(bap_types)
bap_types = bap_types[bap_types >=4]
bap_t = bap_types.index.tolist()
print('bap habitats - ', bap_t)

# number of nvc habitats of interest
nvc_n = len(nvc_t)

print(df_t)

########################################################################
# plotting
########################################################################
'''
for yy in years:

    df_y = df_t[df_t['year'] == yy]

    fig, ax = plt.subplots(dpi = 300    )#figsize = (10,12), ,dpi = 200

    for ii, nn in enumerate(nvc_t):
        df = df_y[df_y['nvc_let'] == nn]
        ax.scatter(df['latitude'], df['longitude'], zorder=1,
            c=colours[ii], alpha = 0.5, label=nn)

    plt.legend(prop={'size': 5})

    ax.set_xlim(map_lat_min, map_lat_max)
    ax.set_ylim(map_long_min, map_long_max)
    plt.axis('off')

    ax.imshow(site_map, extent = box, zorder=0, aspect= 'equal')
    plt.savefig('./figures/nvc_map_'+str(yy)+'.png')
    plt.show()
'''
df_cg = df_t[df_t['nvc_let'] == 'calcicolous grasslands']
cg_subs = df_cg['nvc_num'].unique().tolist()
print(cg_subs)


for yy in years:

    df_y = df_cg[df_cg['year'] == yy]

    fig, ax = plt.subplots(dpi = 300)#figsize = (10,12), ,dpi = 200

    for ii, nn in enumerate(cg_subs):
        df = df_y[df_y['nvc_num'] == nn]
        ax.scatter(df['latitude'], df['longitude'], zorder=1,
            c=colours[ii], alpha = 0.5, label=nn)

    plt.legend(prop={'size': 5})

    ax.set_xlim(map_lat_min, map_lat_max)
    ax.set_ylim(map_long_min, map_long_max)
    plt.axis('off')

    ax.imshow(site_map, extent = box, zorder=0, aspect= 'equal')
    plt.savefig('./figures/cg_map_'+str(yy)+'.png')
    plt.show()

'''
for yy in years:

    df_y = df_t[df_t['year'] == yy]

    fig, ax = plt.subplots(dpi = 300)#figsize = (10,12), ,dpi = 200

    for ii, nn in enumerate(bap_t):
        df = df_y[df_y['bap_broad'] == nn]
        ax.scatter(df['latitude'], df['longitude'], zorder=1,
            c=colours[ii], alpha = 0.5, label=nn)

    plt.legend(prop={'size': 5})

    ax.set_xlim(map_lat_min, map_lat_max)
    ax.set_ylim(map_long_min, map_long_max)
    plt.axis('off')

    ax.imshow(site_map, extent = box, zorder=0, aspect= 'equal')
    plt.savefig('./figures/bap_map_'+str(yy)+'.png')
    plt.show()
'''
