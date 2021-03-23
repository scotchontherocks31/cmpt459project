from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import math
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Model-specific imports
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import xgboost as xgb

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score
import shap

o = os.getcwd()
global data
model_path = o + "\\..\\models\\"

'''
# For Mac/Linux
data = pd.read_csv(o + "/../data/cases_train_processed.csv", parse_dates=True)
model_path = o + "/../models/"
'''
adapath = model_path + 'adaModel.pkl'
xgpath = model_path + 'xgbModel.pkl'
knnpath = model_path + 'knnModel.pkl'

ada_pass = 'ada'
xg_pass = 'xg'
knn_pass = 'knn'

# converts string Series into a cataegorized int series1 for data analysis

def categorize_column(data):
    i = 0
    for value in tqdm(data.unique()):
        data.replace(value, i, inplace=True)
        i += 1
    data = data.apply(pd.to_numeric)
    return data


def build_model(train, val):

    print('XGBoost Model:\n')

    print('1\tSetting training variables')
    dtrain = train.drop(columns='outcome')
    dval = val.drop(columns='outcome')
    dt_label = train['outcome']
    dv_label = val['outcome']

    print('2\tSetting custom XGB DMatrices...')
    xg_dtrain = xgb.DMatrix(dtrain, label=dt_label)
    xg_dval = xgb.DMatrix(dval, label=dv_label)
    xg_test = xgb.DMatrix(dtrain, label=dt_label)

    print('3\tSetting parameters...')
    param = {
        'objective': 'multi:softmax',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'num_parallel_tree': 1,
        'min_child_weight': 1000,
        'gamma': 100,
        'num_class': 4
    }
    evallist = [(xg_dval, 'eval'), (xg_dtrain, 'train')]
    num_round = 500
    early_stopping_rounds = 3

    # XGB training GO!
    print('***\tXGB Initiate\t***')
    xgb_model = xgb.train(param, xg_test, num_round, evallist,
                          early_stopping_rounds=early_stopping_rounds)
    print('***\tCOMPLETE\t***')

    print('4\tExporting model')
    list_pickle = open(xgpath, 'wb')
    pickle.dump(xgb_model, list_pickle)
    list_pickle.close()

    print('Adaboost Model:\n')
    print('***\tADAVBOOST Initiate\t***')
    abc  = AdaBoostClassifier(n_estimators=75, learning_rate=1, algorithm = 'SAMME.R')
    x = train.drop(columns='outcome')
    y = train['outcome']
    model1 = abc.fit(x,y)
    print('***\tCOMPLETE\t***\n')

    list_pickle = open(adapath, 'wb')
    pickle.dump(model1, list_pickle)
    list_pickle.close()
    print('4\tExporting model\n')

    print('KNNboost Model:\n')
    print('***\tKNN Initiate\t***')
    knn_model = KNeighborsClassifier(n_neighbors=1000,p=2,leaf_size=30,weights='uniform')
	model2 = knn_model.fit(x,y)
    print('***\tCOMPLETE\t***\n')
	list_pickle = open(knnpath, 'wb')
	pickle.dump(model2, list_pickle)
	list_pickle.close()
    print('4\tExporting model\n')

    print('All models exported...\n')
    return


def evaluate(train, val, filepath, str):

    # importing models
    print('*****\tEVALUATION INITIATE\t*****')
    model = pickle.load(open(filepath, "rb"))

    dtrain = train.drop(columns='outcome')
    dval = val.drop(columns='outcome')
    dt_label = train['outcome']
    dv_label = val['outcome']

    train_prediction = [round(value) for value in dt_label]  # from training
    prediction = [round(value) for value in dv_label]  # from validation

    print('1\tAccuracy measure:\n')
    model_predict = None
    cm_labels = train['outcome'].unique().tolist()

    if(str == xg_pass):
        print('Setting up XGBoost vars & model prediction...')
        xg_dtrain = xgb.DMatrix(dtrain, label=dt_label)
        xg_dval = xgb.DMatrix(dval, label=dv_label)
        model_predict = model.predict(xg_dval)
        train_accuracy = accuracy_score(
            model.predict(xg_dtrain), train_prediction)
    else:
        model_predict = model.predict(dval)
        train_accuracy = accuracy_score(
            model.predict(dtrain), dt_label)
    print("Train Accuracy: %.2f%%" % (train_accuracy * 100.0))
    accuracy = accuracy_score(model_predict, prediction)
    print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))
    print('2\tConfusion Matrix:\n')
    cm = confusion_matrix(dv_label,model_predict, labels = cm_labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + cm_labels)
    ax.set_yticklabels([''] + cm_labels)
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.show()

    print('3\tPrecision, Recall & F-Score:\n')
    precision = precision_score(dv_label, model_predict, average="macro")
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(dv_label, model_predict, average="macro")
    print('Recall: %f' % recall)
    # f1: tp / (tp + fp + fn)
    f1 = f1_score(dv_label, model_predict, average="macro")
    print('F1 score: %f' % f1)

    if(str != ada_pass):
        print('4\tSHAP Bar Of Importance:\n')
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(dval)
        shap.summary_plot(shap_values, dval, plot_type="bar")

    print('*****\tEVALUATION COMPLETE\t*****')
    return


def main():
    print("\n\nTeam Losers: Milestone 2\n\n")
    print("Modifying data for classifiers...\n")

    data = pd.read_csv( o + "\\..\\data\\cases_train_processed.csv", parse_dates=True)
    #data = pd.read_csv(
    #    o + "/../data/cases_train_processed.csv", parse_dates=True)
    data['sex'] = categorize_column(data['sex'])
    data['outcome'] = categorize_column(data['outcome'])
    data['Combined_Key'] = categorize_column(data['Combined_Key'])

    # handling date column
    # removes columns where date_confirmation has a daterange
    data = data.loc[~data['date_confirmation'].str.contains('-'), :]

    # might need to drop date column if classifiers can't handle it
    data = data.drop(columns=['province', 'country'])
    data = data.drop(columns=['date_confirmation'])

    print("Splitting data into test and validation sets...\n")

    train, val = train_test_split(
        data, test_size=0.2, random_state=69, shuffle=True)

    print("Building training models...\n")
    build_model(train, val)

    print("Model building completed, Evaluating models...\n")

    # -------- Functions for latter parts of the milestone-------------------------
    evaluate(train, val, xgpath, xg_pass)
    evaluate(train, val, adapath, ada_pass)

    # show_overfit(model)

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
