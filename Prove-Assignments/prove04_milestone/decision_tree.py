from sklearn.tree import DecisionTreeClassifier
import numpy as np
from binaryTree import Node
import math

class decisionTreeClassifier():
    def __init__(self):
        pass

    def fit(self, data, targets):
        return decisionTreeModel(data,targets)

class decisionTreeModel():

    def __init__(self,data, targets):
        self.data = data
        self.targets = targets
        self.tree = self.build_tree(data, targets)

    def predict(self, data, targets):
        predictions = []
        for item in data:
            predictions.append(self.predicting(item))
        return predictions

    def predicting(self, data):
        return "predicting"

    def build_tree(self, data, targets):
        data_set = np.unique(data)

        if len(data_set) == 1:
            return Node()
        else: Node()

