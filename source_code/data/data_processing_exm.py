import pandas as pd
import os

# Set working directory to location of the file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Open raw data
df_train = pd.read_json("../../data/raw/train.json")

# make changes (clean...)
no_null = df_train[df_train.isnull()]

# write back to processed folder
no_null.to_json("../../data/processed/output.json")