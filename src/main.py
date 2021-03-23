from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import math
import numpy as np 
import pandas as pd 
import pickle
from sklearn import model_selection

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

	knn_model = KNeighborsClassifier(n_neighbors=1000,p=2,leaf_size=30,weights='uniform')

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

	#leaf_size = list(range(1,50))
	#n_neighbors = list(range(1,30))
	#p=[1,2]#Convert to dictionary
	#hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)#Create new KNN object

	#knn_2 = KNeighborsClassifier()#Use GridSearch
	#clf = model_selection.GridSearchCV(knn_2, hyperparameters, cv=10)#Fit the model

	#print('Loading model fit......\n')

	#X = data.drop(columns=['outcome'])
	#y = data['outcome']

	#best_model = clf.fit(X,y)#Print The value of best Hyperparameters
	#print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
	#print('Best p:', best_model.best_estimator_.get_params()['p'])
	#print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

	#-------- Functions for latter parts of the milestone-------------------------
	#evaluate(train,val)

	#show_overfit(model)

	#KNeighborsClassifier(leaf_size=50, n_neighbors=1000)
	#knn model valid score: 0.711794140985571
	#knn model train score: 0.7094488730906102

	#KNeighborsClassifier(n_neighbors=1000)
	#knn model valid score: 0.711842723194766
	#knn model train score: 0.7093800480156112

	#knn_model = KNeighborsClassifier(n_neighbors=1000,leaf_size=50,p=1)
	#knn model valid score: 0.7119884698223511
	#knn model train score: 0.7094691157597276
	
	#knn_model = KNeighborsClassifier(n_neighbors=1000,leaf_size=50,p=1,weights='distance')
	#knn model valid score: 0.7104014509886479
	#knn model train score: 0.7776180856102962

	#KNeighborsClassifier()
	#knn model valid score: 0.6895596832439961
	#knn model train score: 0.7289385149168228

	#knn_model = KNeighborsClassifier(weights='distance')
	#knn model valid score: 68.7535424527538
	#knn model train score: 75.36507653753193

	#knn_model = KNeighborsClassifier(p=1)
	#knn model valid score: 69.01588638240676
    #knn model train score: 72.91206989388793

    #knn_model = KNeighborsClassifier(n_neighbors=100,p=1,weights='distance')
    #knn model valid score: 70.750271250668
	#knn model train score: 77.69176892588348

	#KNeighborsClassifier(n_neighbors=1000, p=1, weights='distance')
	#knn model valid score: 71.04176450583796
	#knn model train score: 77.76180856102962

	#KNeighborsClassifier(leaf_size=100, n_neighbors=1000, p=1)
	#knn model valid score: 71.2182798659131
	#knn model train score: 70.94164848200224

	#KNeighborsClassifier(leaf_size=100, n_neighbors=1000)
	#knn model valid score: 71.16322002882545
	#knn model train score: 70.92747861362008

	#KNeighborsClassifier(n_neighbors=10)
	#knn model valid score: 68.90900552217778
	#knn model train score: 72.24689578669084

	#knn_model = KNeighborsClassifier(n_neighbors=1000,p=2,leaf_size=30,weights='uniform')
	#knn model valid score: 71.1842723194766
	#knn model train score: 70.93800480156112

	#Overfitting: model.predict(val) < model.predict(train)

	return

if __name__ == '__main__':
    main()
