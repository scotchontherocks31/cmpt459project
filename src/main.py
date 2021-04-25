import numpy as np 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import re
import random
import math
import pandas as pd 
from sklearn import metrics
import matplotlib.pyplot as plt

# Model-specific imports
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

#Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

o = os.getcwd()

data = pd.read_csv(o + "\\..\\data\\cases_train_processed.csv", parse_dates = True)
data_test = pd.read_csv(o + "\\..\\data\\cases_test_processed.csv", parse_dates=True)
plot_path = o + "\\..\\plots\\"
results_path = o + "\\..\\results\\"

'''
# For Mac/Linux
data = pd.read_csv(o + "/../data/cases_train_processed.csv", parse_dates = True)
data_test = pd.read_csv(o + "/../data/cases_test_processed.csv", parse_dates=True)
plot_path = o + "/../plots/"
results_path = o + "/../results/"
'''


# converts string Series into a cataegorized int series for data analysis
def categorize_column(data):
    i = 0
    for value in tqdm(data.unique()):
        data.replace(value, i, inplace = True)
        i += 1
    data = data.apply(pd.to_numeric)
    return data

# converts string Series into a cataegorized int series1 for data analysis
def categorize_outcome(data):
    data = data.map({'nonhospitalized':0, 'deceased':1, 'recovered':2, 'hospitalized':3},na_action ='ignore')
    data = data.apply(pd.to_numeric)
    return data

def cleanup(data):
    data = data.drop(columns=['province', 'country'])
    data['sex'] = categorize_column(data['sex'])
    data['outcome'] = categorize_outcome(data['outcome'])
    data['Combined_Key'] = categorize_column(data['Combined_Key'])
    return data

def main():
    print("\n\nTeam Losers: Milestone 3\n\n")
    print("Modifying data for classifiers...\n")

    data = pd.read_csv( o + "\\..\\data\\cases_train_processed.csv", parse_dates=True)
    #data = pd.read_csv(
    #    o + "/../data/cases_train_processed.csv", parse_dates=True)
    data = cleanup(data)
    data_test = cleanup(data_test)
    
    print("Building training models...\n")
    abc  = AdaBoostClassifier()
    params = {'n_estimators': [75,100], 'learning_rate': [2]}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scoring = {'Overall_Accuracy': metrics.make_scorer(metrics.accuracy_score),
               'Overall_Recall': metrics.make_scorer(metrics.recall_score, average = 'macro'),
               'F1_Deceased': metrics.make_scorer(metrics.f1_score, labels=[1], average= 'micro'),
               'Recall_Deceased': metrics.make_scorer(metrics.recall_score, labels = [1], average = 'micro') }

    X = data.drop(columns='outcome')
    y = data['outcome']

    #n_scores = cross_val_score(abc, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    gs = GridSearchCV(abc, param_grid = params, scoring= scoring, cv=cv, n_jobs=-1, refit = 'Overall_Accuracy')
    gs.fit(X,y)
    print("Model building completed, Evaluating models...\n")

    print(gs.cv_results_)
    
    return

if __name__ == '__main__':
    main()
