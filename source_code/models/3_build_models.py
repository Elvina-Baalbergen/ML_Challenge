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

def model_NLP(df_train,authorId):
    # Prepare training data
    df_train_auth = df_train[df_train['authorId']==authorId] 
    df_train_abstracts = df_train_auth["abstract"].tolist()
    df_train_titles = df_train_auth["title"].tolist()
    df_train_all = df_train_titles + df_train_abstracts

    # build model
    result_NLP = CountVectorizer(input='content')
    result_NLP.fit_transform(df_train_all)
    
    return result_NLP

def model_place(df_train,authorId, estim):
    features = df_train[['year', 'venues_le']]
    target_var = df_train['authorId'] == authorId

    print(target_var)
    
    clf = RandomForestClassifier(n_estimators=estim)
    clf = clf.fit(features, target_var)
    result_place = clf
    return result_place

def build_model(authorId,df_train):
    # calls other model functions
    NLP = model_NLP(df_train, authorId)
    place = model_place(df_train, authorId, 20)
    
    # Stores the built model
    list_models = [NLP,place]

    list_models[0].get_feature_names_out()

    with open(f"../../models/{authorId}.pkl", 'wb') as f:
        pickle.dump(file = f, obj =list_models)

def build_all_models(df_train):
    # for each person in train data, build the models
    authors = df_train['authorId'].unique()

    for author in authors:
        build_model(author,df_train)

def test_place_model():
    set_path()
    authorId = 3188285
    df_train = pd.read_pickle("../../data/processed/train_clean_df.pkl")
    estim = 20

    model = model_place(df_train, authorId, estim)
    print(model.predict_proba([[2014, 58]]))

def test_NLP_model():
    set_path()
    authorId = 3188285	
    df_train = pd.read_pickle("../../data/processed/train_clean_df.pkl")

    df_train_masoud = df_train[df_train['authorId']== authorId]
    df_train_masoud_NLP = df_train_masoud[['title', 'abstract']]
    masoud_sent1 = df_train_masoud_NLP.iloc[0,:].tolist()
    
    model = model_NLP(df_train, authorId)
    
    print(model.transform(masoud_sent1).count_nonzero())

def test_build_model(df_train):
    build_model(3188285,df_train)

    with open(f"../../models/3188285.pkl", 'rb') as f:
        models = pickle.load(file = f)
    
    print(models[0].get_feature_names_out())

set_path()
#df_train = pd.read_pickle("../../data/processed/train_clean_df.pkl")
#build_all_models(df_train)

#test_place_model()
test_NLP_model()