import pandas as pd
import numpy as np
from sklearn import preprocessing
import os

#########
# auth: Elvina
# take in a raw df and output cleaned processed df
# step 2 in the pipeline
##########

def encode_venues(df_train):
    labelencoder = preprocessing.LabelEncoder()
    encoded_labels_venue = labelencoder.fit_transform(df_train['venue'][:].tolist())
    df_train['venues_le'] = encoded_labels_venue
    df_train_venue = df_train.drop(["venue"], axis=1)

    return df_train_venue

# Set working directory to location of the file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Open raw data
df_train = pd.read_pickle("../../data/processed/dirty_df.pkl")

# Call cleaning functions
df_train = encode_venues(df_train)

# write back to processed folder
df_train.to_pickle("../../data/processed/clean_df.pkl")