#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

#cleaning text rows
def process_row(row):
    import re
    from textblob import Word
    from string import punctuation
    from nltk.stem.snowball import SnowballStemmer
    from gensim.parsing.preprocessing import STOPWORDS
    #Mail address
    row = re.sub('(\S+@\S+)(com|\s+com)', ' ', row)
    #Username
    row = re.sub('(\S+@\S+)', ' ', row)
    #punctuation & Lower case
    punctuation = punctuation + '\n' + '—“,”‘-’' #+ '0123456789'
    row = ''.join(word.lower() for word in row if word not in punctuation)
    #Stopwords & Lemma
    stop = STOPWORDS
    row = ' '.join(Word(word).lemmatize() for word in row.split() if word not in stop)
    #Stemming
    stemmer = SnowballStemmer(language='english')
    row = ' '.join([stemmer.stem(word) for word in row.split() if len(word) > 2])
    #Extra whitespace
    row = re.sub('\s{1,}', ' ', row)

    return row

