import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings('ignore')

import clean



with open("./data/file_list", "rb") as fp:
    b = pickle.load(fp)
surveys = [x for x in b if 'Metadata' not in x]

#######################################################################
# Getting the survey data for a specific site
#######################################################################

site_name = 'lulli'
site_files = []
for file_name in surveys:
    if site_name.lower() in file_name.lower():
        site_files.append(file_name)
print(site_files)
print(site_files[-1])

xls = pd.ExcelFile(site_files[1])
s_names = xls.sheet_names
for nn in s_names:
    print(nn)

for name in xls.sheet_names:
    if 'whole' in name.lower():
        wpd_string = name
    if 'species te' in name.lower():
        spec_temp_string = name
    if 'ground' in name.lower():
        ground_string = name

whole = xls.parse(wpd_string)
species = xls.parse(spec_temp_string)
ground = xls.parse(ground_string)

#whole = clean.whole_clean(whole)
#print(whole.head())
#species = clean.species_clean(species)
#print(species.head())
#cover, frequency = clean.get_abund_and_freq(species, column='desc_latin')
#print(cover.head())
ground = clean.ground_clean(ground)
print(ground.head())

#######################################################################
# Getting all survery data
#######################################################################
'''
for nn, ss in enumerate(surveys[:2]):
    print('\n\n\n', nn, '\n\n\n', ss, '\n\n\n')
    xls = pd.ExcelFile(ss)



    whole = xls.parse(wpd_string)
    whole = clean.whole_clean(whole)
    print('\nwhole\n')
    print(whole.head())
    whole = clean.whole_to_ml(whole)
    print('\nwhole for ml\n')
    print(whole.head())

    species = xls.parse(spec_temp_string)
    species = clean.species_clean(species)
    cover, frequency = clean.get_abund_and_freq(species, column='desc_latin')
    print('\ncover\n')
    print(cover.head())
    print('\nfrequency\n')
    print(frequency.head())

    ground = xls.parse(ground_string)
    ground = clean.ground_clean(ground)
    print('\nground\n')
    print(ground.head())
'''