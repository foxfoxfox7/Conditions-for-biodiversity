import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings('ignore')

import clean


with open("./data/file_list", "rb") as fp:
    b = pickle.load(fp)
surveys = [x for x in b if 'Metadata' not in x]

########################################################################
# saving the data in the lookup tab
########################################################################
'''
xls = pd.ExcelFile(surveys[0])
lookup = xls.parse('All_LTMN_Lookups')

lookup = lookup.dropna(how='all', axis = 1)
lookup.columns = lookup.columns.str.lower()

def make_dictionary(df, keys, vals):
    # making a dictionary out of two columns in the dataframe

    df = pd.Series(df[vals].values, index=df[keys])
    return df.dropna().to_dict()

def make_list(df, col):

    values = df[col].values
    cleaned_vals = [val for val in values if str(val) != 'nan']
    return cleaned_vals

lookup_d = {
    'landuse': make_dictionary(lookup, 'landuse_code', 'landuse_type'),
    'dom': make_dictionary(lookup, 'dom_code', 'dom_desc'),
    'slopeform': make_dictionary(lookup, 'slopeform_code', 'slopeform'),
    'speces_desc': make_dictionary(lookup, 'veg_spec', 'desc_latin'),
    'site': make_dictionary(lookup, 'sitecode', 'site'),
    'bap_broad': make_list(lookup, 'bap_broad'),
    'bap_priority': make_list(lookup, 'bap_priority'),
    'feature': make_list(lookup, 'feature')
}

print(lookup_d['bap_broad'])

with open("./my_data/lookup.pkl", "wb") as fp:
    pickle.dump(lookup_d, fp)
'''

########################################################################
# saving a total data dataframe (all the surveys)
########################################################################

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

    # extracting the useful information

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
print(total_data.info())
total_data.to_pickle('./my_data/total_data.pkl')

########################################################################
# saving indicator species for a specific site
########################################################################

site_name = 'lullington'
site_files = []
for file_name in surveys:
    if site_name.lower() in file_name.lower():
        site_files.append(file_name)
print('\n\n', site_files, '\n\n')

# Reverses the order so that it goes in chronological order
site_files = site_files[::-1]

spec_data = []
for site in site_files:
    xls = pd.ExcelFile(site)

    for name in xls.sheet_names:
        if 'species te' in name.lower():
            spec_temp_string = name

    species = xls.parse(spec_temp_string)
    species = clean.species_clean(species)
    cover, frequency = clean.get_abund_and_freq(species, column='desc_latin')

    spec_data.append(cover)

n_sites = len(data)

df_spec = pd.concat(spec_data)
df_spec.columns = df_spec.columns.str.lower()
spec_l = df_spec.columns.tolist()

print(spec_l)
print(len(spec_l))

clean.check_names(spec_l)

# Creating the dataframe with the indicator species data

df_spec_t = pd.DataFrame(index=spec_l)

# cover % maintained in heathland
df_spec_t['h bryophytes and lichens'] = None

# at least two present in heathland
df_spec_t['h dwarf shrubs'] = None

# Total Ulex and/or Genista spp. cover <50%, with Ulex europaeus <25%
# heath
df_spec_t['h ulex or genista'] = None

# cg3+ and cg2 40-90%
df_spec_t['cg non-graminae'] = None

# heathland. 1 frequent and 2 occasional
df_spec_t['h graminoids'] = None

# cg3 positive indicators (Bromopsis erecta (if CG3) or Brachypodium
# pinnatum (if CG4 ), or both (if CG5) frequent plus at least two
# species/taxa frequent and four occasional throughout the  sward)
df_spec_t['cg3+ +ve'] = None

# cg2 positive indicators (At least four species/taxa frequent plus at
# least three species occasional throughout the sward.)
df_spec_t['cg2 +ve'] = None

# heathland desirable forbs (At least 2 species at least occasional
# throughout the sward)
df_spec_t['h forbs'] = None

# heathland negative exotic species (<1%)
df_spec_t['h exotic species'] = None

# heathland negative (occasional)
df_spec_t['h Acrocarpous mosses'] = None

# heathland negative (<10%)
df_spec_t['h bracken'] = None

#cg3+ negative (<5%)
df_spec_t['cg3 -ve 5'] = None

# cg2 -ve (<10%)
df_spec_t['cg3 -ve 10'] = None

#cg2 negative (<5%)
df_spec_t['cg2 -ve 5'] = None

#cg2 negative (<10%)
df_spec_t['cg2 -ve 10'] = None

# heathland negative herbaceous species (<1%)
df_spec_t['h herbaceous'] = None

# cg3 tree and scrub species excluding juniperus communis
# (no more than 5%)
df_spec_t['cg tree scrub'] = None

# heathland negative species tree and scrub (<15%)
df_spec_t['h tree scrub'] = None

print(df_spec_t.head())
