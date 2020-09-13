import pandas as pd
import numpy as np
import pickle
import re
import csv

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
# saving the data in nvc species lists
########################################################################

xls = pd.ExcelFile('./docs/NVC-floristic-tables.xls')
print(xls.sheet_names)
df_flortab = xls.parse('NVCfloristictables (26.3.2008)')

# rearranging the data so we get a column for each habitat and a list
# of the species beneath it
df_s_piv = df_flortab.pivot(columns = 'Community or sub-community code',
    values = 'Species name or special variable ')
all_cols = df_s_piv.columns.tolist()

#keep_cols = [col for col in all_cols if re.search(r'\d+$', col)]
drop_cols = [col for col in all_cols if not re.search(r'\d+$', col)]
df_s_piv = df_s_piv.drop(drop_cols, axis=1)

# doing the same for the typical cover percentages
df_c_piv = df_flortab.pivot(columns = 'Community or sub-community code',
    values = 'Species constancy value')
df_c_piv = df_c_piv.drop(drop_cols, axis=1)

# doing the same for the maximum abundances
df_a_piv = df_flortab.pivot(columns = 'Community or sub-community code',
    values = 'Maximum abundance of species')
df_a_piv = df_a_piv.drop(drop_cols, axis=1)

# getting rid of all the nans as well as putting the species list
# at the top of the column
final_cols = []
for nvc in df_s_piv.columns:
    df = pd.DataFrame()
    print('extracting', nvc, 'columns\n')

    # THE SPECIES COLUMN

    # gets rid of all the NaNs that would be present when combining
    spec_l = []
    for i in df_s_piv[nvc].iteritems():
        # only unique values and not nan
        if str(i[1]) != 'nan':
            spec_l.append(i[1])
    sp_l_len = len(spec_l)
    df[nvc] = spec_l

    # THE COVER COLUMN

    cov_l = []
    for i in df_c_piv[nvc].iteritems():
        # only unique values and not nan
        if str(i[1]) != 'nan':#!= 'nan'
            cov_l.append(i[1])
    cov_l_len = len(cov_l)

    # the species list is longest as the bottom ones are physical
    # characteristics whihc dont have cover or abundance vals
    for i in range(sp_l_len - cov_l_len):
        cov_l.append(np.NaN)
    df[nvc+'_[c]'] = cov_l

    # THE ABUNDANCE COLUMN

    abund_l = []
    for i in df_a_piv[nvc].iteritems():
        # only unique values and not nan
        if str(i[1]) != 'nan':
            abund_l.append(i[1])
    abund_l_len = len(abund_l)

    for i in range(sp_l_len - abund_l_len):
        abund_l.append(np.NaN)
    df[nvc+'_[a]'] = abund_l

    final_cols.append(df)

df_cg_final = pd.concat(final_cols, axis = 1)

print(df_cg_final)

#print(df_cg_final)
#print(df_cg_final.info())
#print(df_cg_final['A1'].dropna().tolist())

with open("./my_data/nvc_spec.pkl", "wb") as fp:
    pickle.dump(df_cg_final, fp)

########################################################################
# saving a total data dataframe (all the surveys)
########################################################################
'''
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
'''
########################################################################
# saving indicator species for a specific site
########################################################################
'''
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

df_spec = pd.concat(spec_data)
df_spec.columns = df_spec.columns.str.lower()
spec_l = df_spec.columns.tolist()

print(spec_l)
print(len(spec_l))

clean.check_names(spec_l)

df_sp_save = pd.DataFrame()
df_sp_save['spec_list'] = spec_l

#with open('./Lullington/Lullington_sp.csv', 'w', newline='') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(spec_l)

df_sp_save.to_csv('./Lullington/Lullington_sp.csv', index = False)
'''
