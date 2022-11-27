import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle

#########
# auth: Elvina
# make prediction of the models
# step 4 in the pipeline
##########

# Set working directory to location of the file
def set_path():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

def prediction(year, venue, title, abstract):
    # prepare test cases 
    testcase_NLP = title + " " + abstract
    testcase_place = [year, venue]

    # Get all author names
    set_path()
    df_train = pd.read_pickle("../../data/processed/clean_df.pkl")
    authors = df_train['authorName'].unique()

    # Loop over all authors
    result_records = []

    for i in range(len(authors)):
        author = authors[i]
        
        # Open the authors model
        with open(f"../../models/{author}.pkl", 'rb') as f:
                models = pickle.load(file = f)
        
        # calculate NLP Score
        NLP_model = models[0]
        Score_nlp = score_NLP(NLP_model,testcase_NLP)

        # Calculate place Score
        place_model = models[1]
        Score_place = score_place(place_model, testcase_place)

        # Calculate combined Score
        total_predictions = calculate_final_score(Score_nlp, Score_place)

        # add results to records
        result_records.append([author,Score_nlp,Score_place,total_predictions])

        #Log
        #print(f"{i} - {Score_nlp},{Score_place}")

    # make dataframe from recors
    df_result = pd.DataFrame(result_records, columns = ["author","nlp_score","place_score","total_score"])
    
    # get highest score
    sorted = df_result.sort_values("total_score",ascending=False)
    print(sorted.head(10))

    authorName = sorted.head(1).author.tolist()[0]
    return authorName

def score_NLP(model, testcase):
    modelcount = len(model.get_feature_names_out())
    testcount = model.transform([testcase]).count_nonzero()
    score = testcount / modelcount
    return score

def score_place(model, testcase):
    score = model.predict_proba([testcase])[0][1]
    return score

def calculate_final_score(score_NLP,score_place):
    final_score = score_NLP * score_place
    return final_score

def test_prediciton():
    set_path()
    df_train = pd.read_pickle("../../data/processed/clean_df.pkl")
    df_test = df_train[df_train['authorName']=='Chuhan Wu']
    year = df_test["year"].tolist()[0]
    venue = df_test["venues_le"].tolist()[0]
    title = df_test["title"].tolist()[0]
    abstract = df_test["abstract"].tolist()[0]

    import warnings
    warnings.filterwarnings('ignore')

    print(prediction(year,venue,title,abstract))

test_prediciton()