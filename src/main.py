import pandas as pd 
import numpy as np 
import os
import math

o = os.getcwd()
train_data = pd.read_csv(o + "\\..\\data\\cases_train.csv", parse_dates = True)
test_data = pd.read_csv(o + "\\..\\data\\cases_test.csv", parse_dates = True)
location_data= pd.read_csv(o + "\\..\\data\\location.csv", parse_dates = True)

def clean_database(data):

	return data

def remove_outlers(data, column):
	mean = data[column].mean()
	std = data[column].std()

	cutoff = std * 3
	lower_bound, upper_bound = mean + std, mean - std
	for index, row in data:
		if row[column] > upper_bound or row[column] <= lowerbound:
			data.drop(index, axis=0, inplace=True)
	return

def handle_outliers(data):
	return data

def main():

	return

if __name__ == '__main__':
    main()