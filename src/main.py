import pandas as pd 
import numpy as np 
import os
import re
import math

o = os.getcwd()
train_data = pd.read_csv(o + "\\..\\data\\cases_train.csv", parse_dates = True)
test_data = pd.read_csv(o + "\\..\\data\\cases_test.csv", parse_dates = True)
location_data= pd.read_csv(o + "\\..\\data\\location.csv", parse_dates = True)

def clean_database(data):

	return data

def gaussian_remove_outliers(data, column):
	mean = data[column].mean()
	std = data[column].std()

	cutoff = std * 3
	lower_bound, upper_bound = mean + cutoff, mean - cutoff
	for index, row in data:
		if row[column] > upper_bound or row[column] <= lowerbound:
			data.drop(index, axis=0, inplace=True)
	return

def handle_outliers(data):
	return data

def clean(data):
	#remove unnecessary columns
	data.drop(['source', 'additional_information'], axis=1, inplace=True)

	#remove rows containing NaN values with minimal loss to data
	data.dropna(subset['longitude', 'latitude', 'province', 'date_confirmation'], inplace=True)

	#standardize age formatting
	for index, row in data.iterrows():
		if row['age'] != np.NaN:
			m = re.match("(\d{1,})-(\d{1,})", str(row['age']))
			if m:
				data.loc[index,'age'] = round(sum(map(int, m.groups()))/2)
				n = re.match("(\d{1,})\W",str(row['age']))
			if n:
				data.loc[index,'age'] = m.groups()[0]

	return data	


def main():
	print("\n\nTeam Losers: Milestone 1\n\n")
	print("Performing Data Cleaning and Handling of NaN Values...")

	print("Cleaning Train Data...")
	cleaned_train = clean(train_data)

	print("Cleaning Test Data...")
	cleaned_test = clea(test_data)


	return

if __name__ == '__main__':
    main()