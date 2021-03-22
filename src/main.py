from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import math
import numpy as np 
import pandas as pd 
import pickle

# Model-specific imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

o = os.getcwd()
global data 
'''
model_path = o + "\\..\\models\\"
'''
model_path = o + "/../models/"
knnpath = model_path + 'knnModel.pkl'


#For Mac/Linux
#data = pd.read_csv(o + "/../data/cases_train_processed.csv", parse_dates = True)
#model_path = o + "/../models/"

# converts string Series into a cataegorized int series1 for data analysis
def categorize_column(data):
	i = 0
	for value in tqdm(data.unique()):
		data.replace(value, i, inplace = True)
		i += 1
	data = data.apply(pd.to_numeric)
	return data

def build_model(data):

	X = data.drop(columns=['outcome'])
	y = data['outcome']

	knn_model = KNeighborsClassifier(n_neighbors=10)

	model2 = knn_model.fit(X,y)

	list_pickle = open(knnpath, 'wb')
	pickle.dump(model2, list_pickle)
	list_pickle.close()
	#save model to model_path + "model_name.pkl"
	print("Second model saved to folder...\n")

	return

def evaluate(train, val):
	
	#importing models
	#AdaBoost
	return

def main():
	print("\n\nTeam Losers: Milestone 2\n\n")
	print("Modifying data for classifiers...\n")

	data = pd.read_csv(o + "/../data/cases_train_processed.csv", parse_dates = True)
	data['sex'] = categorize_column(data['sex'])
	data['outcome'] = categorize_column(data['outcome'])
	data['Combined_Key'] = categorize_column(data['Combined_Key'])

	#handling date column
	#removes columns where date_confirmation has a daterange
	data = data.loc[~data['date_confirmation'].str.contains('-'), :]  

	#might need to drop date column if classifiers can't handle it
	data = data.drop(columns=['province', 'country'])
	data = data.drop(columns=['date_confirmation'])

	print("Splitting data into test and validation sets...\n")

	train, val = train_test_split(data, test_size=0.2, random_state=69, shuffle=True)

	print("Building second model...\n")	
	build_model(train)

	print("Model building completed, Evaluating models...")


	#-------- Functions for latter parts of the milestone-------------------------
	#evaluate(train,val)

	#show_overfit(model)

	return

if __name__ == '__main__':
    main()
