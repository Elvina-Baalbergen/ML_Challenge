import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle
import warnings
import multiprocessing 

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

def true_auth_score(authorId,year, venue, title, abstract):
    with open(f"../../models/{authorId}.pkl", 'rb') as f:
            models = pickle.load(file = f) 

    # prepare test cases 
    testcase_NLP = title + " " + abstract
    testcase_place = [year, venue]

    # calculate NLP Score
    NLP_model = models[0]
    Score_nlp = score_NLP(NLP_model,testcase_NLP)

    # Calculate place Score
    place_model = models[1]
    Score_place = score_place(place_model, testcase_place)

    # Calculate combined Score
    total_predictions = calculate_final_score(Score_nlp, Score_place)

    return (Score_nlp,Score_place,total_predictions)



def prediction(year, venue, title, abstract):
    # prepare test cases 
    testcase_NLP = title + " " + abstract
    testcase_place = [year, venue]

    # Get all author names
    set_path()
    df_train = pd.read_pickle("../../data/processed/test_clean_df.pkl")
    authors = df_train['authorId'].unique()

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
        #print(f"{i} - {Score_nlp},{Score_place}", end="\r")

    # make dataframe from recors
    df_result = pd.DataFrame(result_records, columns = ["author","nlp_score","place_score","total_score"])
    
    # get highest score
    sorted = df_result.sort_values("total_score",ascending=False)
    print(sorted.head(10))


    authorId = sorted.head(1).author.tolist()[0]
    return authorId

def score_NLP(model, testcase):
    modelcount = len(model.get_feature_names_out())
    testcount = model.transform([testcase]).count_nonzero()
    score = testcount / modelcount
    return score

def score_place(model, testcase):
    score = model.predict_proba([testcase])[0][1]
    return score

def calculate_final_score(score_NLP,score_place):
    final_score = score_NLP #* score_place
    return final_score

def test_prediciton():
    set_path()
    df_train = pd.read_pickle("../../data/processed/test_clean_df.pkl")
    df_test = df_train[df_train['authorName']=='Chuhan Wu']
    year = df_test["year"].tolist()[0]
    venue = df_test["venues_le"].tolist()[0]
    title = df_test["title"].tolist()[0]
    abstract = df_test["abstract"].tolist()[0]

    print(prediction(year,venue,title,abstract))

def predict_all():
    # Setup predicitons dfs
    set_path()
    df_predictions = pd.read_pickle("../../data/processed/test_clean_df.pkl")
    
    # Log 
    print("Running predicitons on test set:")

    # Generate prediciton for each testcases
    df_predictions["predicted_auth"] = df_predictions.head(20).apply(lambda x: prediction(x.year,x.venues_le,x.title,x.abstract), axis=1)

    with open(f"../../data/testing/predictions.pkl", 'wb') as f:
        pickle.dump(file = f, obj =df_predictions)

def predict_all_parrallel(path_testcases):
    # Setup predicitons dfs
    set_path()
    df_test= pd.read_pickle(path_testcases)

    # Log 
    print("Running predicitons on test set:")

    # open results file to write to
    f_result =  open("../../data/testing/results.csv", "a")
    f_result.write("index,predicted_auth,real_author,correct\n")
    f_result.flush()

    # Set up Process pool - no arg automatically matches computers corecount
    ProcPool = multiprocessing.Pool(1) 

    for result in ProcPool.imap(predict_map,df_test.iterrows()):
        log = f"{result[0]},{result[1]},{result[2]},{result[1] == result[2]}"
        f_result.write(log +"\n")
        f_result.flush()
        print(log)

    f_result.close()
    ProcPool.close()

def predict_map(df_iter):
    index, series =df_iter

    year = series['year']
    venue = series['venues_le']
    title = series['title']
    abstract = series['abstract']

    predicted_author = prediction(year, venue, title, abstract)
    real_author = series['authorId']

    print(true_auth_score(real_author,year, venue, title, abstract))

    return [index,predicted_author,real_author]

warnings.filterwarnings('ignore')
predict_all_parrallel("../../data/processed/test_clean_df.pkl")