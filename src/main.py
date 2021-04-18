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
locations_path = o + "\\..\\results\\location_transformed.csv"
'''
#For Mac/Linux

train_data = pd.read_csv(o + "/../data/cases_train.csv", parse_dates = True)
test_data = pd.read_csv(o + "/../data/cases_test.csv", parse_dates = True)
location_data= pd.read_csv(o + "/../data/location.csv", parse_dates = True)
test_processed_path = o + "/../results/cases_test_processed.csv"
train_processed_path = o + "/../results/cases_train_processed.csv"
locations_path = o + "/../results/location_transformed.csv"
'''

#takes in a dataset containing all the cities in a state and returns a single row with the aggregated data
def aggregate_states(data):
	lat, lon = round(data['Lat'].mean(), 6), round(data['Long_'].mean(), 6)
	t_confirmed = data['Confirmed'].sum()
	t_deaths = data['Deaths'].sum()
	t_recovered = data['Recovered'].sum()
	t_active = data['Active'].sum()
	combined_key = data.iloc[0]['Province_State'] + ', US'
	incidence_rate = data['Incidence_Rate'].max()
	case_fat_ratio = round((t_deaths / (t_confirmed - 1 ))*100, 6)

	summary = [(data.iloc[0]['Province_State'], 'United States', data.iloc[0]['Last_Update'], lat, lon,
	            t_confirmed, t_deaths, t_recovered, t_active, combined_key, incidence_rate, case_fat_ratio)]

	row = pd.DataFrame(summary, columns = list(data.columns))

	return row

def transform_locations(data):
	#create dataframe of just the United States locations
	US = data.loc[data['Country_Region'] == 'US']

	#create a list of States/ Provinces in US
	states = list(US['Province_State'].unique())

	New_US = data[0:0]

	#iterates through each state, groups the cities together and aggregates them
	for state in states:
		cities = US.loc[US['Province_State'] == state]
		state_row = aggregate_states(cities)
		New_US = New_US.append(state_row, ignore_index=True)

	new_data = data.drop(index= list(US.index.values))
	new_data = new_data.append(New_US, ignore_index=True)

	new_data = new_data.sort_values(by=['Country_Region', 'Province_State'])
	return new_data


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
	data.dropna(subset=['longitude', 'latitude', 'country'], inplace=True)

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
    
    #imputing missing date values based on mode\
	mode = data['date_confirmation'].mode()
	data['date_confirmation'].fillna(mode, inplace=True)

	#imputing categorical sex values based on random values
	#list_sex = data['sex'].dropna().values
	#data['sex'] = data['sex'].apply(lambda x: np.random.choice(list_sex))

	#imputing categorical sex values based on mode
	data['sex'].fillna(data['sex'].dropna().mode()[0], inplace=True)

	return data	

def join_datasets_onlatlong(train, location):
	#one-liner join code
	join = pd.merge(train, location, how='inner', left_on=[
					"latitude", "longitude"], right_on=["Lat", "Long_"])
	join.drop(['Lat', 'Long_', 'Last_Update', 'Province_State', 'Country_Region'], axis=1, inplace=True)
	join.sort_values(by=["country", "province"], ascending=True, inplace=True)

	#printing out missing value statistics on the join
	print("Join statistics:\n")
	j_na = join.isna().sum()
	j_total = len(join)
	print("Number of missing Values for Join:\n", j_na)
	print("\n")
	print("Percentage of missing values for Join:\n", round(j_na/j_total, 2))
	print("\n")

	#return the joined dataset
	return join

def join_datasets_onprovcount(train, location):
	#one-liner join code
	join = pd.merge(train, location, how='left', left_on=[
	                "province", "country"], right_on=["Province_State", "Country_Region"])
	join.drop(['date_confirmation','Lat', 'Long_', 'Last_Update', 'Province_State', 'Country_Region'], axis=1, inplace=True)
	join.sort_values(by=["country", "province"], ascending=True, inplace=True)

	#fill in missing values
	join.country.replace(['Czech Republic','Democratic Republic of the Congo','Puerto Rico','Reunion', 'South Korea','Republic of Congo'],['Czechia','Congo (Kinshasa)','United States','France', 'Korea, South','Congo (Brazzaville)'],inplace=True)
	join.province.replace(['San Juan'],['Puerto Rico'], inplace=True)
	for index, row in tqdm(join.iterrows()):
		if math.isnan(row['Confirmed']):
			filter = locations_tf['Combined_Key'] == row['country']
			query = locations_tf[filter]
			print(row.country, row.province)
			try:
				join.loc[index,'Confirmed'] = query['Confirmed'].values[0]
			except IndexError:
				query = locations_tf.loc[(locations_tf.Country_Region == row['country']) & (locations_tf.Province_State == 'Unknown')]
				try:
					join.loc[index,'Confirmed'] = query['Confirmed'].values[0]
				except IndexError:
					query = locations_tf.loc[(locations_tf.Country_Region == row['country']) & (locations_tf.Province_State == row['province'])]
					try:
						join.loc[index,'Confirmed'] = query['Confirmed'].values[0]
					except IndexError:					
						filter = locations_tf['Country_Region'] == row['country']
						query = locations_tf[filter]
						join.loc[index,'Confirmed'] = query['Confirmed'].mean().round()
						join.loc[index,'Deaths'] = query['Deaths'].mean().round()
						join.loc[index,'Recovered'] = query['Recovered'].mean().round()
						join.loc[index,'Combined_Key'] = query['Country_Region'].values[0]
						join.loc[index,'Active'] = query['Active'].mean().round()
						join.loc[index,'Incidence_Rate'] = query['Incidence_Rate'].mean().round()
						join.loc[index,'Case-Fatality_Ratio'] = query['Case-Fatality_Ratio'].mean().round()
						continue
			join.loc[index,'Deaths'] = query['Deaths'].values[0]
			join.loc[index,'Recovered'] = query['Recovered'].values[0]
			join.loc[index,'Combined_Key'] = query['Combined_Key'].values[0]
			join.loc[index,'Active'] = query['Active'].values[0]
			join.loc[index,'Incidence_Rate'] = query['Incidence_Rate'].values[0]
			join.loc[index,'Case-Fatality_Ratio'] = query['Case-Fatality_Ratio'].values[0]
	#printing out missing value statistics on the join
	print("Join statistics:\n")
	j_na = join.isna().sum()
	j_total = len(join)
	print("Number of missing Values for Join:\n", j_na)
	print("\n")
	print("Percentage of missing values for Join:\n", round(j_na/j_total, 2))
	print("\n")

	#return the joined dataset
	return join

def main():
	print("\n\nTeam Losers: Milestone 1\n\n")
	print("Performing Data Cleaning and Handling of NaN Values...\n")

	print("Cleaning Train Data...\n")
	cleaned_train = clean(train_data)	

	print("Cleaning Test Data...\n")
	cleaned_test = clean(test_data)

	print("\nRemoving Outliers from Train Data...\n")
	processed_train = handle_outliers(cleaned_train)

	print("\nTransforming Locations Dataset...\n")
	locations_tf = transform_locations(location_data
)	print("Saving Location Data to file...\n")	
	locations_tf.to_csv(locations_path, index=False)

	#print("Producing a Join on Train & Location Data using (latitude, longitue)...\n")
	#lat_long_test = join_datasets_onlatlong(processed_train, locations_tf)
	#lat_long_test.to_csv(o + "/../results/lat_long_test.csv", index=False)

	print("Producing a Join on Train & Location Data using (province, country)...\n")
	prov_coun_test = join_datasets_onprovcount(processed_train, locations_tf)
	print("Saving Train Data to file...\n")
	prov_coun_test.to_csv(train_processed_path, index=False)

	print("Producing a Join on Test & Location Data using (province, country)...\n")
	prov_coun_test = join_datasets_onprovcount(cleaned_test, locations_tf)
	print("Saving Test Data to file...\n")
	prov_coun_test.to_csv(test_processed_path, index=False)

	return

if __name__ == '__main__':
    main()
