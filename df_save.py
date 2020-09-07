import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings('ignore')

import clean


with open("./data/file_list", "rb") as fp:
    b = pickle.load(fp)
surveys = [x for x in b if 'Metadata' not in x]

data = []

for nn, ss in enumerate(surveys):
    print('\n\n\n', nn, '\n\n\n', ss, '\n\n\n')
    xls = pd.ExcelFile(ss)

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

    data.append(df)


total_data = pd.concat(data)

print(total_data.head())
print(total_data.info())

total_data.to_pickle('total_data.pkl')
