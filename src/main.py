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
test_processed_path = o + "\\..\\results\\cases_test_processed.csv"
train_processed_path = o + "\\..\\results\\cases_train_processed.csv"

#For Mac/Linux
'''
train_data = pd.read_csv(o + "/../data/cases_train.csv", parse_dates = True)
test_data = pd.read_csv(o + "/../data/cases_test.csv", parse_dates = True)
location_data= pd.read_csv(o + "/../data/location.csv", parse_dates = True)
test_processed_path = o + "/../results/cases_test_processed.csv"
train_processed_path = o + "/../results/cases_train_processed.csv"
'''

def gaussian_remove_outliers(data, column):
	mean = data[column].mean()
	std = data[column].std()
	cutoff = std * 3
	lower_bound, upper_bound = mean - cutoff, mean + cutoff

	lam, lom = data['latitude'].mean(), data['longitude'].mean()
	lasd, losd = data['latitude'].std(), data['longitude'].std()
	cutla, cutlo = lasd*2, losd*2
	lbla, lblo = lam - cutla, lom - cutlo
	ubla, ublo = lam + cutla, lom + cutlo

	for index, row in tqdm(data.iterrows()):
		if row[column] > upper_bound or row[column] <= lower_bound:
			data.drop(index, axis=0, inplace=True)
			continue

		if (row['longitude'] > ublo or row['longitude'] <= lblo) and (row['latitude'] > ubla or row['latitude']<= lbla):
			data.drop(index, axis=0, inplace=True)

	return data

def handle_outliers(data):

	#handle oultier ages
	data = gaussian_remove_outliers(data, 'age')
	#data = geog_remove_outliers(data)

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

	print("Cleaning Test Data...\n")
	cleaned_test = clean(test_data)
	cleaned_test.to_csv(test_propcessed_path, index=False)

	print("\nRemoving Outliers from Train Data\n")
	processed_train = handle_outliers(cleaned_train)
	processed_train.to_csv(train_processed_path, index=False)

	print("Outliers Removed")

	return

if __name__ == '__main__':
    main()