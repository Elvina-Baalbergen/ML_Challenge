import pandas as pd
import os

#########
# auth: Elvina
# take data and stores it as dataframe
# step 1 in the pipeline
##########

# Set working directory to location of the file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Open raw data
df_train = pd.read_json("../../data/raw/train.json")

# write back to processed folder
df_train.to_pickle("../../data/processed/dirty_df.pkl")