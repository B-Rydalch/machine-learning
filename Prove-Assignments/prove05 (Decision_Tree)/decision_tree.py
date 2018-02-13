import math
import numpy as np
import pandas as pd
import entropy as enpy
from binary_tree import Node


class Decision_Tree_Classifier():

    def __init__(self):
        pass

    def fit(self, data, targets, headers):
        return Decision_Tree_Model(data, targets, headers)

class Decision_Tree_Model():
    def __init__(self, data, targets, headers):
        self.data= data
        self.targets = targets
        self.headers = headers
        frames = [data, targets]
        train_data = pd.concat(frames, axis=1)
        self.tree = enpy.build_tree(train_data, headers[:-1])

    def __repr__(self):
        enpy.print_tree(self.tree)

    def predict(self, data):
        predictions = []
        enpy.print_tree(self.tree)
        for index, row in data.iterrows():
            attributes = self.headers[:]
            prediction = self.predict_class(self.tree, row, attributes)
            predictions.append(prediction)
        return predictions

    def predict_class(self, node, row_data, headers):
        for attribute in headers:
            if attribute == node.name:
                value = row_data[attribute]
                if value in node.children:
                    new_node = node.children[value]
                    if new_node.isLeaf():
                        return new_node.name
                    else:
                        headers.remove(attribute)
                        return self.predict_class(new_node, row_data, headers)
                else:
                    most_common = enpy.most_common(self.targets)
                    return most_common
        return 1

