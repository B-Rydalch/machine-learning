import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from knn import kNeighborsClassifier
from decision_tree import decisionTreeClassifier
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from binary_tree import Node


def main():

    print("Select which file to import. \n \
    1. UCI Car Evaluation \n \
    2. Pima Indian Diabetes \n \
    3. Automobile MPG \n \
    4. Credit Screening")

    user_input = int(input("> "))

    if (user_input == 1):
        train_data, test_data = build_car_dataset()
        classifier = kNeighborsClassifier()

    elif (user_input == 2):
        train_data, test_data = build_diabetes_dataset()
        classifier = kNeighborsClassifier()

    elif (user_input == 3):
        train_data, test_data = build_mpg_dataset()
        classifier = kNeighborsClassifier()

    elif (user_inpu == 4):
        train_data, test_data = build_credit_screen_dataset()
        classifier = decisionTreeClassifier()
        


    x = train_data
    y = test_data
    kf = KFold(n_splits=4)
    kf.get_n_splits(x,y)
    KFold(n_splits=4, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(x, y):
        #x_train = train_data, x_test = test_data
        #y_train = train_targets, y_test = test_targets

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    #select the kNearest Neighbor
    classifier = kNeighborsClassifier()
    model = classifier.fit(x_train, y_train)
    predictions = model.predict(x_test)

    #perform prediction to test data 
    targets_predicted = model.predict(y_test)

    count = 0 
    for index in range(len(x_test)):
        if targets_predicted[index] == test_data[index]:
            count += 1

    #provide accuracy percentage
    correctness = float(count) / len(test_data) * 100 
    print "Accuracy: {:.2f}".format(correctness)


def build_car_dataset():

    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "target"]
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", header=None, names=headers)
    df = df.sample(frac = 1).reset_index(drop=True)


    # object for handling non-numerical data
    replacementObject = {
        'buying': {
            'vhigh': 3,
            'high': 2,
            'med': 1,
            'low': 0
        },

        'maint': {
            'vhigh': 3,
            'high': 2,
            'med': 1,
            'low': 0
        },

        'doors': {
            '5more': 3,
            '4': 2,
            '3': 1,
            '2': 0
        },

        'persons': {
            'more':2,
            '4': 1,
            '2': 0
        },

        'lug_boot':{
            'big': 2,
            'med': 1,
            'small': 0
        },

        'safety':{
            'high': 2,
            'med': 1,
            'low': 0
        },

        'target':{
            'vgood': 3,
            'good':2,
            'acc': 1,
            'unacc':0
        }
    }

    # handle the missing data.
    df.replace(replacementObject,inplace=True)

    train_data = df.as_matrix(headers[0:6])
    test_data = df.as_matrix(headers[6:7])

    return train_data, test_data


def build_diabetes_dataset():

    headers = ["# preg", "plasma glucose", "Diastolic BP",
    "Triceps SKT", "serum insulin", "BMI", "Diabetes pedigree"
    "Age", "Class v"]

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data", header=None, names=headers)

    df = df.sample(frac=1).reset_index(drop=True)
    train_data = df.as_matrix(headers[0:7])
    test_data = df.as_matrix(headers[7:8])

    # handle the missing data.
    df.replace(0, np.NaN ,inplace=True)
    df.dropna(inplace=True)

    return train_data, test_data

def build_mpg_dataset():

    headers = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", header=None, names=headers, na_values='?')
    df = df.sample(frac=1).reset_index(drop=True)
    
  #sam  df = df.pop('car name')
    counter = -1
    # for index in range(headers):
    #     counter += 1
    #     print "header # ",counter, " ", headers[index], 
    
    
    train_data = df.as_matrix(headers[1:8])
    test_data = df.as_matrix(headers[0:1]).astype(float)

    print "test: ", test_data
    # handle the missing data.
    #df.replace('?', -1, inplace=True)

    return train_data, test_data



def build_credit_screen_dataset(data, classes, feature):
    headers = ["credit", "income", "collateral", "should_loan"]
    dataset = read_csv('loan.csv', delimiter = ',', header = None, names = headers)

    train_data = dataset.as_matrix(headers[0:3])
    test_data = dataset.as_matrix(headers[3:4])

    return train_data, test_data, headers




if __name__ == "__main__":
    main()
