#
# Author : Marcos Teixeira
# SkyNet is watching you
#


# common imports


import numpy as np
import pandas as pd

import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import matplotlib.pyplot as plt
import lightgbm as lgb


def linear_regression_experiment(xtrain, xtest, ytrain, ytest):
    # baseline approach : Linear Regression using all variables
    from sklearn.linear_model import LogisticRegression

    # building the model
    model = LogisticRegression()
    model.fit(xtrain, ytrain)

    preds = model.predict(xtest)

    accuracy = accuracy_score(preds,ytest)
    recall = recall_score(preds,ytest)
    precision = precision_score(preds,ytest)
    f1 = f1_score(preds,ytest)

    print("accuracy : {}".format(accuracy))
    print("recall : {}".format(recall))
    print("precision : {}".format(precision))
    print("f1 score : {}".format(f1))
    # accuracy : 0.9994666666666666
    # recall : 1.0
    # precision : 0.68
    # f1 score : 0.8095238095238095

def lightGBM_experiment(xtrain, xtest, ytrain, ytest, columns, plot_importance=False):

    # parameters for LightGBMClassifier
    params = {
        'objective' :'multiclass',
        'learning_rate' : 0.02,
        'num_leaves' : 31,
        'is_unbalance': 'true',
        "max_depth": -1,
        "num_class": 2,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'boosting_type' : 'gbdt',
        'verbosity': 1
    }

    lgtrain = lgb.Dataset(xtrain,ytrain)
    clf = lgb.train(params, lgtrain, 300,feature_name=list(columns))
    preds = clf.predict(xtest)
    preds = np.argmax(preds, axis=1)

    accuracy = accuracy_score(preds,ytest)
    recall = recall_score(preds,ytest)
    precision = precision_score(preds,ytest)
    f1 = f1_score(preds,ytest)

    print("accuracy : {}".format(accuracy))
    print("recall : {}".format(recall))
    print("precision : {}".format(precision))
    print("f1 score : {}".format(f1))
    # accuracy : 0.9996666666666667
    # recall : 0.9545454545454546
    # precision : 0.84
    # f1 score : 0.8936170212765958
    if plot_importance:
        ax = lgb.plot_importance(clf)
        ax.plot()
        plt.show()


def NN_experiment(xtrain, xtest, ytrain, ytest, plot_importance=True):
    # baseline approach : Linear Regression using all variables
    from sklearn.neural_network import MLPClassifier

    # building the model
    model = MLPClassifier(hidden_layer_sizes=(200, ), activation='relu', solver='adam', alpha=0.0001,
    batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200,
     shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
     nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
     epsilon=1e-08, n_iter_no_change=10)
    model.fit(xtrain, ytrain)

    preds = model.predict(xtest)

    accuracy = accuracy_score(preds,ytest)
    recall = recall_score(preds,ytest)
    precision = precision_score(preds,ytest)
    f1 = f1_score(preds,ytest)

    print("accuracy : {}".format(accuracy))
    print("recall : {}".format(recall))
    print("precision : {}".format(precision))
    print("f1 score : {}".format(f1))
    # accuracy : 0.9996333333333334
    # recall : 0.9333333333333333
    # precision : 0.84
    # f1 score : 0.8842105263157894



# paths
DATASITH_PATH='/Users/marcostexeira/Downloads/DESAFIO_CREDITO/'
DATASITH_FILE='desafio_fraude.csv'


def load_fraud_data(data_path,file):
    csv_path = os.path.join(data_path, file)
    return pd.read_csv(csv_path)

# loading data
dataset = load_fraud_data(DATASITH_PATH,DATASITH_FILE)
np_dataset = dataset.values

# data split
xtrain, xtest, ytrain, ytest = train_test_split(np_dataset[:, :-1],np_dataset[:, -1],test_size=0.2, random_state=42)

ytrain = ytrain.astype(int)
ytest = ytest.astype(int)

lightGBM_experiment(xtrain, xtest, ytrain, ytest, dataset.columns[:-1].values,True)
