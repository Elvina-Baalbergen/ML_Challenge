import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle

#########
# auth: Elvina
# building models and ensemble them in a pipeline
# step 3 in the pipeline
##########

# Set working directory to location of the file
def set_path():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

def model_NLP(df_train,authname):
    # Prepare training data
    df_train_auth = df_train[df_train['authorName']==authname] 
    df_train_abstracts = df_train_auth["abstract"].tolist()
    df_train_titles = df_train_auth["title"].tolist()
    df_train_all = df_train_titles + df_train_abstracts

    # build model
    result_NLP = CountVectorizer(input='content')
    result_NLP.fit_transform(df_train_all)
    
    return result_NLP

def model_place(df_train,authname, estim):
    features = df_train[['year', 'venues_le']]
    target_var = df_train['authorName'] == authname
    
    clf = RandomForestClassifier(n_estimators=estim)
    clf = clf.fit(features, target_var)
    result_place = clf
    return result_place

def build_model(authname):
    # reads raw data
    set_path()
    df_train = pd.read_pickle("../../data/processed/clean_df.pkl")

    # calls other model functions
    NLP = model_NLP(df_train, authname)
    place = model_place(df_train, authname, 20)
    
    # Stores the built model
    list_models = [NLP,place]

    list_models[0].get_feature_names_out()

    with open(f"../../models/{authname}.pkl", 'wb') as f:
        pickle.dump(file = f, obj =list_models)

def build_all_models(df_train):
    # for each person in train data, build the models
    authors = df_train['authorName'].unique()

    for author in authors:
        build_model(author)

def test_place_model():
    set_path()
    authname = 'Chuhan Wu'
    df_train = pd.read_pickle("../../data/processed/clean_df.pkl")
    estim = 20

    model = model_place(df_train, authname, estim)
    print(model.predict_proba([[2021, 119]]))

def test_NLP_model():
    set_path()
    authname = 'Chuhan Wu'
    df_train = pd.read_pickle("../../data/processed/clean_df.pkl")

    df_train_ryan = df_train[df_train['authorName']=='Ryan Cotterell']
    df_train_ryan_NLP = df_train_ryan[['title', 'abstract']]
    ryan_sent1 = df_train_ryan_NLP.iloc[6,:].tolist()
    
    model = model_NLP(df_train, authname)
    
    print(model.transform(ryan_sent1).count_nonzero())

def test_build_model():
    build_model("Chuhan Wu")

    with open(f"../../models/Chuhan Wu.pkl", 'rb') as f:
        models = pickle.load(file = f)
    
    print(models[0].get_feature_names_out())

set_path()
df_train = pd.read_pickle("../../data/processed/train_clean_df.pkl")
build_all_models(df_train)
