import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../..')

import clean


# Creating the original with the indicator species data only
#df = pd.read_csv('Lullington_sp.csv')#, index_col=0

# reading in the indicator ecel file after edits
xls = pd.ExcelFile('indicator_form.xlsx')
for name in xls.sheet_names:
    print(name)
df = xls.parse('Sheet1')

species_n = df['spec_list'].unique()
#print(species_n)

#clean.check_names(species_n, 80)

#clean.replace_matches_in_column(df, 'spec_list', 'ulex europaeus')
#clean.replace_matches_in_column(df, 'spec_list', 'rubus fruticosus agg')

def find_in_spec(df, indicators, col_name = 'dummy'):

    # so that it doesn't overwite data already in the column
    if col_name not in df.columns:
        df[col_name] = None

    indicators_lower = [sp.lower() for sp in indicators]

    for i in df['spec_list'].iteritems():
        for name in indicators_lower:
            if name in i[1]:
                #print(i[1])
                df.loc[i[0], col_name] = 1

########################################################################
#
# entering data from fct forms and matching species list
#
########################################################################
'''
print(df.info())

# bryophytes and lichens
#(cover % maintained in heathland)
bryophytes_lichens = []
find_in_spec(df, bryophytes_lichens, '(h)_bryophytes_lichens')
print(df.info())

# Total Ulex and/or Genista spp. heath
#cover <50%, with Ulex europaeus <25%
ulex_genista = ['ulex', 'Genista']
find_in_spec(df, ulex_genista, '(h)_ulex_genista')

# non graminae cg3+ and cg2
# (40-90%)
non_graminae = []
find_in_spec(df, non_graminae, '(cg)_non-graminae')

# dwarf shrubs in heathland
# at least two present
dw_shrubs = ['Arctostaphylos uva-ursi', 'Calluna vulgaris', 'Empetrum nigrum',
    'Erica ciliaris', 'Eucalyptus cinerea', 'Erica tetralix', 'Erica vagans',
    'Genista anglica', 'Genista pilosa', 'Ulex gallii', 'Ulmus minor',
    'Vaccinium myrtillus', 'Vaccinium vitis-idaea']
find_in_spec(df, dw_shrubs, '(h) dwarf shrubs')

# graminoids heathland.
# (1 frequent and 2 occasional)
h_graminoids = ['Agrostis', 'Ammophila arenaria', 'Carex',
    'Danthonia decumbens', 'Deschampsia flexuosa', 'Festuca',
    'Molinia caerulea', 'Nardus stricta', 'Trichophorum cespitosum']
find_in_spec(df, h_graminoids, '(h) graminoids')

# cg3 positive indicators
#(Bromopsis erecta (if CG3) or Brachypodium
# pinnatum (if CG4 ), or both (if CG5) frequent plus at least two
# species/taxa frequent and four occasional throughout the  sward)
cg3_pos = ['Brachypodium pinnatum', 'Bromopsis erecta', 'Anthyllis vulneraria',
    'Asperula cynanchica', 'Campanula glomerata', 'Centaurea scabiosa',
    'Cirsium acaule', 'Filipendula vulgaris', 'Galium verum',
    'Genista tinctoria', 'Gentianella', 'Helianthemum nummularium',
    'Hippocrepis comosa', 'Leontodon hispidus', 'Littorina saxatilis',
    'Leucanthemum vulgare', 'Linum catharticum', 'Lotus corniculatus',
    'Onobrychis viciifolia', 'Pilosella officinarum', 'Plantago media',
    'Polygala', 'Primula veris', 'Sanguisorba minor',
    'Scabiosa columbaria', 'Serratula tinctoria', 'Succisa pratensis',
    'Viola hirta', 'Thymus']
find_in_spec(df, cg3_pos, '(cg)3_pos')

# cg2 positive indicators
# (At least four species/taxa frequent plus at
# least three species occasional throughout the sward.)
cg2_pos = ['Anthyllis vulneraria', 'Asperula cynanchica',
    'Campanula glomerata', 'Cirsium acaule', 'Filipendula vulgaris',
    'Genista tinctoria', 'Gentianella', 'Helianthemum nummularium',
    'Hippocrepis comosa', 'Leontodon hispidus', 'Littorina saxatilis',
    'Leucanthemum vulgare', 'Linum catharticum', 'Lotus corniculatus',
    'Pilosella officinarum', 'Plantago media', 'Polygala',
    'Primula veris', 'Sanguisorba minor', 'Scabiosa columbaria',
    'Serratula tinctoria', 'Succisa pratensis', 'Thymus']
find_in_spec(df, cg2_pos, '(cg)2_pos')

# heathland desirable forbs
# (At least 2 species at least occasional throughout the sward)
forbs = ['Armeria maritima', 'Galium saxatile', 'Genista anglica',
    'Hypochaeris radicata', 'Lotus corniculatus', 'Plantago lanceolata',
    'Plantago maritima', 'Polygala serpyllifolia', 'Potentilla erecta',
    'Rumex acetosella', 'Scilla verna', 'Serratula tinctoria',
    'Thymus praecox', 'Viola riviniana']
limestone_forbs = ['Filipendula vulgaris', 'Galium verum',
    'Helianthemum nummularium', 'Sanguisorba minor']
find_in_spec(df, forbs, '(h) forbs')

# heathland negative exotic species
# (<1%)
exotic = ['Rhododendron ponticum', 'Gaultheria shallon', 'Fallopia japonica']
find_in_spec(df, exotic, '(h) exotic species')

# Acrocarpous mosses heathland negative
# (occasional)
acro_mosses = ['Campylopus introflexus']
find_in_spec(df, acro_mosses, '(h) Acrocarpous mosses')

# bracken heathland negative
# (<10%)
bracken = ['Pteridium aquilinum']
find_in_spec(df, bracken, '(h) bracken')

#cg3 negative
# (<5%)
cg3_neg5 = ['Cirsium arvense', 'Cirsium vulgare', 'Rumex crispus',
    'Rumex obtusifolius', 'Senecio jacobaea', 'Urtica dioica']

find_in_spec(df, cg3_neg5, '(cg)3_neg5')

# cg3 -ve
# <10%
cg3_neg10 = ['Brachypodium pinnatum']
find_in_spec(df, cg3_neg10, '(cg)3_neg10')

#cg2 negative (<10%)
# <10%
cg2_neg10 = ['Brachypodium pinnatum', 'Bromopsis erecta']
find_in_spec(df, cg2_neg10, '(cg)2_neg10')

# heathland negative herbaceous species
# (<1%)
# (excluding. Epilobium. palustre)
# ‘coarse grasses’
herbacous_neg = ['Cirsium arvense', 'Digitalis purpurea', 'Epilobium',
    'Chamerion angustifolium', 'Juncus effusus', 'Juncus squarrosus',
    'Ranunculus', 'Senecio', 'Rumex obtusifolius', 'Urtica dioica',
    'Urtica dioica', 'Jacobaea vulgaris', 'Cirsium']
find_in_spec(df, herbacous_neg, '(h) herbaceous')

# cg3 tree and scrub species excluding juniperus communis
# (no more than 5%)
tree_scrub = ['Betula', 'Prunus spinosa', 'Pinus', 'Rubus',
    'Sarothamnus scoparius', 'Quercus', 'Hippophae rhamnoides']
find_in_spec(df, tree_scrub, '(cg) tree scrub')

# heathland negative species tree and scrub
# (<15%)
find_in_spec(df, tree_scrub, '(cg) tree scrub')
'''
########################################################################
#
# printing out new dataframe
#
########################################################################

#print(df.head())
print(df.info())

df.to_excel('indicator_form.xlsx', index = False)
cols = df.columns.tolist()

# converts the 1s in the df to the names of the species
df_spec = pd.DataFrame()
for cc in cols:
    df_spec[cc] = df.loc[df[cc].notna(), 'spec_list']

# turns all the non NaN values in a series to a list
def get_species(df, col):

    spec_l = df[col].tolist()
    return [ss for ss in spec_l if str(ss) != 'nan']

# create a dictionary witht he lists of indicator species
indicator_d = {}
for cc in cols:
    indicator_d[cc] = get_species(df_spec, cc)

with open("./indicator_d.pkl", "wb") as fp:
    pickle.dump(indicator_d, fp)
