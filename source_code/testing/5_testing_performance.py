import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle

#########
# auth: Elvina
# make predictions and test the performance of the models NLP and Place
# step 4 in the pipeline
##########

# Set working directory to location of the file
def set_path():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)