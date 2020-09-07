import pickle
import numpy as np
import pandas as pd

import clean


with open("./data/file_list", "rb") as fp:
    b = pickle.load(fp)
surveys = [x for x in b if 'Metadata' not in x]

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
