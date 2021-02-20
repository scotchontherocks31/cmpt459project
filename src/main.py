import numpy as np 
from tqdm import tqdm
import os
import re
import random
import math
import pandas as pd 


o = os.getcwd()
train_data = pd.read_csv(o + "\\..\\data\\cases_train.csv", parse_dates = True)
test_data = pd.read_csv(o + "\\..\\data\\cases_test.csv", parse_dates = True)
location_data= pd.read_csv(o + "\\..\\data\\location.csv", parse_dates = True)

def gaussian_remove_outliers(data, column):
	mean = data[column].mean()
	std = data[column].std()

	cutoff = std * 3
	lower_bound, upper_bound = mean + cutoff, mean - cutoff
	for index, row in data:
		if row[column] > upper_bound or row[column] <= lower_bound:
			data.drop(index, axis=0, inplace=True)
	return

def handle_outliers(data):
	return data

def clean_age(data): #handles specific imputations of age data
	data['age'].interpolate(inplace=True)
	#replace any remaining ages with the average
	mean = data['age'].sum()/ len(data['age'])
	data['age'].fillna(mean, inplace=True)
	#round the ages to whole ints
	data['age'] = data['age'].round().astype(int)

	return data	

def clean(data):
	#remove unnecessary columns
	data.drop(['source', 'additional_information'], axis=1, inplace=True)

	#remove rows containing NaN values with minimal loss to data
	data.dropna(subset=['longitude', 'latitude', 'province', 'country','date_confirmation'], inplace=True)

	#standardize age formatting
	for index, row in tqdm(data.iterrows()):
		if row['age'] is not np.NaN:

			#catch 15-35
			m = re.match("(\d{1,})-(\d{1,})", str(row['age']))
			if m:
				data.loc[index,'age'] = round(sum(map(int, m.groups()))/2)
				continue			
			
			#catch 80+
			n = re.match("(\d{1,})\W", str(row['age']))
			if n:
				#print(row['age'])
				data.loc[index,'age'] = n.groups()[0]
				continue
	#fills in missing ages using linear interpolation// might want to look into sklearn imputers
	data.to_csv( "temp.csv", index=False)
	filtered_data = pd.read_csv('temp.csv')
	data = clean_age(filtered_data)

	#remove temp file
	os.remove('temp.csv')

	#imputing categorical sex values based on random values
	list_sex = data['sex'].dropna().values
	data['sex'] = data['sex'].apply(lambda x: np.random.choice(list_sex))

	#imputing categorical sex values based on mode
	#data['sex'].fillna(data['sex'].dropna().mode()[0], inplace=True)

	return data	


def main():
	print("\n\nTeam Losers: Milestone 1\n\n")
	print("Performing Data Cleaning and Handling of NaN Values...\n")

	print("Cleaning Train Data...\n")
	cleaned_train = clean(train_data)
	cleaned_train.to_csv(o + "\\..\\results\\cases_train_processed.csv", index=False)

	print("Cleaning Test Data...\n")
	cleaned_test = clean(test_data)
	cleaned_test.to_csv(o + "\\..\\results\\cases_test_processed.csv", index=False)

	return

if __name__ == '__main__':
    main()