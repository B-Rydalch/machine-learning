import pandas as pd
from neuron import Neuron
from network_layer import Network_Layer
from sklearn import datasets
from pimaIndiansDiabetes import PIMA_INDIANS_DIABETES


def main(): 
    
    print("Select which dataset to import. \n \
    1. Iris\n \
    2. Pima Indian Diabetes)

    user_input1 = int(input("> "))

    print("Enter thresshold value ")
    user_input2 = (input("> "))
   
    thresh_hold = user_input2

    if (user_input == 1):
        Iris = datasets.load_iris()
        neural_net = Network_Layer(3,4,0)

    elif (user_input == 2):
        train_data, test_data = build_diabetes_dataset()

    neuron1 = Neuron(2, thresh_hold)

    print("testing neuron")
    neuron1.input(1)
    print (neuron1)


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


if __name__ == "__main__":
    main()