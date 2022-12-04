import pandas as pd
import numpy as np
from sklearn import preprocessing
import os

import re
from textblob import Word
from string import punctuation as pn
from nltk.stem.snowball import SnowballStemmer
from gensim.parsing.preprocessing import STOPWORDS

#########
# auth: Elvina / Ulviyya
# take in a raw df and output cleaned processed df
# step 2 in the pipeline
##########

def encode_venues(df_train):
    labelencoder = preprocessing.LabelEncoder()
    encoded_labels_venue = labelencoder.fit_transform(df_train['venue'][:].tolist())
    df_train['venues_le'] = encoded_labels_venue
    df_train_venue = df_train.drop(["venue"], axis=1)

    return df_train_venue

#cleaning text rows
def process_row(row):
    #Mail address
    row = re.sub('(\S+@\S+)(com|\s+com)', ' ', row)
    #Username
    row = re.sub('(\S+@\S+)', ' ', row)
    #punctuation & Lower case
    punctuation = pn + '\n' + '—“,”‘-’' #+ '0123456789'
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

def split_test_train(df_to_split):
    # get a set of authors which appear minimum 2 times on the training data
    df_auth_freq = df_to_split.groupby(['authorId']).size().reset_index().sort_values(0,ascending=False)
    df_auth_freq.rename(columns = {0:'freq'}, inplace=True)
    df_auth_freq_min2papers = df_auth_freq[ df_auth_freq['freq'] >= 2]
    authorset = set( df_auth_freq_min2papers['authorId'].to_list())

    # select only part of the df containing these authors
    mask = []
    for i in range(len(df_train)):
        mask.append(df_train['authorId'][i] in authorset)
    df_subset_auth = df_train[mask]

    # Set up test set
    df_subset_test = df_subset_auth[0:0]

    # Split the test / train set
    df_subset_auth = df_subset_auth.reset_index()
    df_subset_test_len = 0

    for i in range(len(df_subset_auth)):
        if (df_subset_auth['authorId'][i] in authorset):
            # for logging
            print(f"{i} - {df_subset_auth['authorId'][i]}")
            
            # Copy over row to test set
            df_subset_test.loc[df_subset_test_len] = df_subset_auth.loc[i]
            df_subset_test_len +=1

            # author no longer needed in auth set because already a paper copied
            authorset.discard(df_subset_auth['authorId'][i])

            #drop row from training set
            df_subset_auth = df_subset_auth.drop(i)

    return df_subset_auth, df_subset_test

# Set working directory to location of the file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Open raw data
df_train = pd.read_pickle("../../data/processed/dirty_df.pkl")

# Call cleaning functions
print("Encoding venues")
df_train = encode_venues(df_train)
print("Cleaning titles")
df_train['title'] = df_train['title'].apply(process_row)
print("Cleaning abstracts")
df_train['abstract'] = df_train['abstract'].apply(process_row)

# Split into training and testing data
df_train_split, df_test_split = split_test_train(df_train)

# write back to processed folder
df_train_split.to_pickle("../../data/processed/train_clean_df.pkl")
df_test_split.to_pickle("../../data/processed/test_clean_df.pkl")