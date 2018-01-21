from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from knn import kNeighborsClassifier
from sklearn.model_selection import train_test_split


#possible Extras:
#KD-Tree
#Experiment with several more datasets
#Handle non-numeric data





def main():
    build_dataset()


def build_dataset():

    #load the data set
    iris = datasets.load_iris()

    #returns a numpy array for each variable, this will allow us to use the variables to test our algorithm
    data_train, data_test, targets_train, test_target = train_test_split(iris.data, iris.target, test_size = .3)

    #select the kNearest Neighbor
    classifier = kNeighborsClassifier()
    model = classifier.fit(data_train, targets_train)
    predictions = model.predict(data_test)

    #perform prediction to test data 
    targets_predicted = model.predict(data_test)

    count = 0 
    for index in range(len(data_test)):
        if targets_predicted[index] == test_target[index]:
            count += 1

    #provide accuracy percentage
    correctness = float(count) / len(data_test) * 100 
    print "Accuracy: {:.2f}".format(correctness)

if __name__ == "__main__":
    main()
