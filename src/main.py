import numpy as np 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import re
import random
import math
import pandas as pd 

o = os.getcwd()

data = pd.read_csv(o + "\\..\\data\\cases_train_processed.csv", parse_dates = True)
model_path = o + "\\..\\models\\"

'''
#For Mac/Linux
data = pd.read_csv(o + "/../data/cases_train_processed.csv", parse_dates = True)
model_path = o + "/../models/"
'''
def build_model(data):

	#Model code here!!

	# save model to model_path + "model_name.pkl"
	return

def main():
	print("\n\nTeam Losers: Milestone 2\n\n")
	print("Splitting data into test and validation sets...\n")

	train, val = train_test_split(data, test_size=0.2, random_state=69, shuffle=True)

	print("Building first model...\n")	
	build_model(train)

	print("Model building completed, Evaluating models...")
	#-------- Functions for latter parts of the milestone-------------------------
	#evaluate(model,train,val)

	#show_overfit(model)

	return

if __name__ == '__main__':
    main()
