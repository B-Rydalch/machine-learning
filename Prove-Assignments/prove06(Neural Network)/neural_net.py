from sklearn import datasets
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from network_layer import NeuralNetworkCalssifier


# prompts the user for their desired data set 
def get_data_set():
    print("CALLED get_data_set \n")
    while True:
        print("What dataset would you like to work with?")
        print("1 - Iris data set")
        print("2 - Pima Indian Diabetes data set")


        user_input = int(input("> "))

        if user_input == 1:
            data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
            columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]
            data.columns = columns
            for col in columns:
                data[col] = data[col].astype("category")

            return data, "Iris", False

        elif user_input == 2:
            data = pd.read_csv(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/"
                "pima-indians-diabetes.data")
            data.columns = ["pregnancies", "glucose", "blood pressure", "tricep thickness", "insulin", "bmi",
                            "pedigree", "age", "diabetic"]
            data["diabetic"].replace([0, 1], ["non-diabetic", "diabetic"], inplace=True)
            data["diabetic"] = data["diabetic"].astype("category")
            data.replace(0, np.NaN, inplace=True)
            data.dropna(inplace=True)
            return data, "Pima Indian Diabetes", False

        else:
            print("Not a valid input.")

# gets how many times to test
def get_num_tests():
    print("CALLED get_num_tests \n")
    
    is_number = False
    k = 3
    
    while not is_number or k <= 2:
        print("How many tests do you want to run?")
        if k <= 2:
            print("Must be more than 2 tests.")

        is_number = True
        
        # handles non-integer values
        try:
            k = int(input("> "))
        except:
            print("You must enter a number!")
            is_number = False

    return k


#  gets the classifier the user wants 
#  to test with. 
def get_classifier(is_regressor_data):
    print("Called get_classifier \n")
    
    # prompt user
    print("Which algorithm would you like to use?")
    
    while True:
        print("1 - scikit-learn Gaussian")
        print("2 - Hard Coded Neural Network Classifier")
        print("3 - scikit-learn Neural Network Classifier")

        chosen_algorithm = int(input("> "))

        if chosen_algorithm == 1:
            return GaussianNB(), "Gaussian"
        elif chosen_algorithm == 2:
            return NeuralNetworkCalssifier(), "Hard Coded Neural Network Classifier"
        elif chosen_algorithm == 3:
            return MLPClassifier(), "scikit-learn Neural Network Classifier"
        else:
            print("Not a valid user_input.")

# gets multiple classifiers in case the user wants 
# to compare the data against various tests.
def get_multiple_classifiers(is_regressor_data):
    print("Called get_mult_classifier")
    
    # prompt user for which algorithm
    classifiers = dict()
    print("Which algorithms would you like to test?")
    print("(Enter an algorithm one at a time, pressing enter after each addition)")
    user_input = ""

    while user_input != "done" and user_input != "Done":
        print("1 - scikit-learn Gaussian")
        print("2 - Hard Coded Nerual Network Classifier")
        print("3 - scikit-learn Neural Network Classifier")
        print("Type \"done\" when completed.")
        
        user_input = int(input("> "))
        
        if user_input == 1:
            classifiers["scikit-learn Gaussian"] = GaussianNB()
        elif user_input == 2:
            classifiers["Hard Coded Neural Network Classifier"] = NeuralNetworkCalssifier()
        elif user_input == 3:
            classifiers["scikit-learn Neural Network Classifier"] = MLPClassifier()
        elif user_input != "Done" or user_input != "done":
            print("Not a valid user_input.")

    while user_input != "done" or user_input != "Done":
        print("1 - scikit-learn Gaussian")
        print("2 - Hard Coded Nerual Network Classifier")
        print("3 - scikit-learn Neural Network Classifier")
        print("Type \"done\" when completed.")
        
        user_input = int(input("> "))
        
        if user_input == 1:
            classifiers["scikit-learn Gaussian"] = GaussianNB()
        elif user_input == 2:
            classifiers["Hard Coded Neural Network Classifier"] = NeuralNetworkCalssifier()
        elif user_input == 3:
            classifiers["scikit-learn Neural Network Classifier"] = MLPClassifier()
        elif user_input != "Done" or user_input != "done":
            print("Not a valid user_input.")

    return classifiers

# cleans/categorizes data
def clean(data):
    print("CALLED clean")
   
    # categorizes the columns that are nominal
    non_numeric_cols = data.select_dtypes(["category"]).columns

    # replaces all nominal data with numerical values
    data[non_numeric_cols] = data[non_numeric_cols].apply(lambda x: x.cat.codes)

    # sets the values to their z-score. 
    # this allows the values have the same weight
    return data.apply(zscore)

# tests the algorithm k times using k-cross validation
def k_cross_validation(data_set, algorithm, k, needs_cleaned):
    print("CALLED k_cross_validation \n")
    
    kf = KFold(n_splits=k)
    sum = 0

    # randomize the data
    data_set = data_set.sample(frac=1)

    # split data k times and test
    for train, test in kf.split(data_set):
        
        # grab training and testing data
        train = data_set.iloc[train]
        test = data_set.iloc[test]

        # grab test accuracy
        accuracy = test_data(train, test, algorithm, needs_cleaned)
        sum += accuracy

    # return average accuracy
    return sum / k

# tests data on a chosen algorithm
def test_data(train, test, algorithm, needs_cleaned):
    print("Called test_data \n")
    
    # separates the data and puts it into a numpy array
    if needs_cleaned:
        train_data = np.array(clean(train[train.columns[0: -1]]))
        test_data = np.array(clean(test[test.columns[0: -1]]))
    
    else:
        train_data = np.array(train[train.columns[0: -1]])
        test_data = np.array(test[test.columns[0: -1]])

    train_target = np.array(train.iloc[:, -1])
    test_target = np.array(test.iloc[:, -1])

    # train & test the algorithm
    model = algorithm.fit(train_data, train_target)
    targets_predicted = model.predict(test_data)

    # count num of correct predictions
    count = 0
    for index in range(len(targets_predicted)):
        if targets_predicted[index] == test_target[index]:
            count += 1

    # return algorithm accuracy
    return 0.0 if count == 0 else count / len(targets_predicted)

# prints the accuracy of a given data set
def print_accuracy(classifier_name, data_set_name, accuracy):
    print("CALLED print_accuracy \n")
    print("\nThe " + classifier_name + " was " + str(round(accuracy * 100, 3)) +
          "% accurate on the " + data_set_name + " Data Set.")

def test_algorithm():
    print("CALLED test_algorithm \n")

    # get the data set,classifier, number of tests
    data_set, data_set_name, is_regressor_data = get_data_set()
    classifier, classifier_name = get_classifier(is_regressor_data)
    k = get_num_tests()

    # gets the accuracy of the the algorithm on the data set
    accuracy = k_cross_validation(data_set, classifier, k, classifier_name)

    # display accuracy
    print_accuracy(classifier_name, data_set_name, accuracy)

# compares multiple algorithms for user convenience 
def compare_algorithms():
    print("CALLED Compare_algorithms \n")

    # get the data set
    data_set, data_set_name, is_regressor_data = get_data_set()

    # get the classifiers the user wants to compare
    classifiers = get_multiple_classifiers(is_regressor_data)

    # get number of times the user wants to run the classifier on the data
    k = get_num_tests()

    # displays all of the classifier's accuracy
    for classifier_name in classifiers.keys():
        accuracy = k_cross_validation(data_set, classifiers[classifier_name], k,
                                      classifier_name != "Decision Tree Classifier")
        print_accuracy(classifier_name, data_set_name, accuracy)

def main():
    print("Called Main \n")
    print("What would you like to do?")
    print("1 - Test an algorithm")
    print("2 - Compare multiple algorithms")
    
    user_input = int(input("> "))

    if user_input == 1:
        test_algorithm()
    elif user_input == 2:
        compare_algorithms()
    else:
        print("ERROR! invalid input")

if __name__ == "__main__":
    main()