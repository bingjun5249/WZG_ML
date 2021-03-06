import numpy as np
import pandas as pd

eee_infile = "eee_binary.h5"
eem_infile = "eem_binary.h5"
emm_infile = "emm_binary.h5"
mmm_infile = "mmm_binary.h5"

eee_df = pd.read_hdf(eee_infile)
eem_df = pd.read_hdf(eem_infile)
emm_df = pd.read_hdf(emm_infile)
mmm_df = pd.read_hdf(mmm_infile)

print(eee_df)
print(eem_df)
print(emm_df)
print(mmm_df)

print(eee_df.columns)
print(eem_df.columns)
print(emm_df.columns)
print(mmm_df.columns)
