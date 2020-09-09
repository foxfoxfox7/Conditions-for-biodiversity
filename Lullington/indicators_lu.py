import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings('ignore')

import clean


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
