import numpy as np
import random as rn
import operator
from math import exp
import matplotlib.pyplot as plt

class NeuralNetworkCalssifier:
    def __init__(self):
        self.num_nodes = []
        self.num_hidden_layers = int(input("With how many hidden layers?\n> "))
        for i in range(self.num_hidden_layers):
            self.num_nodes.append(int(input("How many nodes in layer " + str(i + 1) + "?\n> ")))

    def fit(self, data, target):
        num_rows, num_cols = data.shape        
        neural_network = Neurons(num_cols, self.num_hidden_layers, self.num_nodes, target)
        neural_network.teach(data)

        return NeuralNetworkModel(neural_network)


# the node which holds the weights between a neuron and the targets
class Target_Node:
    def __init__(self, num_inputs, target=None):
        self.input_weights = []
        self.target = target
        self.value = 0
        self.delta = 0
        self.bias = -1

        # assign random weight values
        for _ in range(num_inputs + 1):
            self.input_weights.append(rn.uniform(-1, 1))

    # sigmoid function to determine whether or not the neuron fires
    @staticmethod
    def sigmoid(value):
        return 1 / (1 + exp(-value))

    # trains to have correct weights
    def train(self, data_row):
        self.value = 0
        
        # gets the sum of the weights times the data input
        for index in range(len(data_row)):
            self.value += data_row[index] * self.input_weights[index + 1]

        # add the biased node
        self.value += self.bias * self.input_weights[0]
        self.value = self.sigmoid(self.value)

        return self.value


class Neurons:
    def __init__(self, num_cols, num_hidden_layers, num_nodes, targets):
        self.neural_network = [[] for _ in range(num_hidden_layers + 1)]
        self.targets = targets
        self.unique_targets = set(targets)
        self.num_hidden_layers = num_hidden_layers
       
        if num_hidden_layers > 0:
            # create first layer 
            for node in range(num_nodes[0]):
                self.neural_network[0].append(Target_Node(num_cols))

            # create the hidden layers
            for index in range(num_hidden_layers - 1):
                for _ in range(num_nodes[index + 1]):
                    self.neural_network[index + 1].append(Target_Node(len(self.neural_network[index])))

            # create the output layer
            for unique_target in self.unique_targets:
                self.neural_network[num_hidden_layers].append(Target_Node(
                    len(self.neural_network[num_hidden_layers - 1]), unique_target))

        else:  # No hidden layers, only input and output
            for unique_target in self.unique_targets:
                self.neural_network[0].append(Target_Node(num_cols, unique_target))

    # teaches the neuron array when to fire when given data
    def teach(self, data):
       
        # this will run when either all the weights are correct or after 1000 runs
        done = False
        runs = 0
        accuracy = []
        ac = self.get_accuracy(data)
        accuracy.append(ac)
        
        print("Starting accuracy: " + str(round(ac * 100, 3)) + "\n")
        
        if self.num_hidden_layers > 0:
            
            while not done and runs < 1000:
                # everything was predicted correctly
                done = True
               
                # teach counter
                runs += 1
                
                # loop through each row of data
                for index, data_row in enumerate(data):
                    
                    # 2D array to keep track of nodes values at each layer
                    if self.num_hidden_layers > 1:
                        hidden_node_values = [[] for _ in range(self.num_hidden_layers)]
                    else:
                        hidden_node_values = [[]]
                    
                    # set up the first layer with the data as inputs
                    for node in self.neural_network[0]:
                        hidden_node_values[0].append(node.train(data_row))
                    
                    # set up all the hidden ayers with the previous layer's activation as the value
                    for layer_index, layer in enumerate(self.neural_network[1:-1]):
                        for node in layer:
                            hidden_node_values[layer_index + 1].append(node.train(hidden_node_values[layer_index]))
                   
                    # a dictionary with the target name as the key and its activation as the value
                    target_values = dict()
                    
                    # gets the activation for each target
                    for node in self.neural_network[-1]:
                        target_values[node.target] = node.train(hidden_node_values[-1])
                    
                    # get the target with the highest activation value
                    prediction = min(target_values, key=target_values.get)#self.get_key_with_max_value(target_values)
                    
                    # if the highest activation value was the correct target, it predicted correctly!
                    if self.targets[index] != prediction:
                        self.recalculate_node_values(prediction, self.targets[index], data_row)
                        done = False
                accuracy.append(self.get_accuracy(data) * 100)

        else:  # No hidden layers
            
            # runs either 1000 times or if it guesses every target correctly
            while not done and runs < 1000:
                # if this never changes, everything was predicted correctly
                done = True
                
                # tests counter
                runs += 1
                
                # loop through each row of data
                for index, data_row in enumerate(data):
                    # a dictionary with the target name as the key and its activation as the value
                    target_values = dict()
                    
                    # loop through each node in the neural network and calculate its activation
                    for node in self.neural_network[0]:
                        target_values[node.target] = node.train(data_row)
                    
                    # get the target with the highest activation value
                    prediction = min(target_values, key=target_values.get)#self.get_key_with_max_value(target_values)
                    
                    # if the highest activation value was the correct target, it predicted correctly!
                    if self.targets[index] != prediction:
                        self.recalculate_node_values(prediction, self.targets[index], data_row)
                        
                        # if did not guess correctly, we're going to have to loop again.
                        done = False
                accuracy.append(self.get_accuracy(data) * 100)
        print("TEST DONE")
        print("ENDING ACCURACY: " + str(round(accuracy[-1], 3)))
        x = range(0, len(accuracy), 1)
        plt.close()
        plt.plot(x, accuracy)
        plt.show()

    # recalculate 
    def recalculate_node_values(self, wrongly_predicted_target, correct_target, data):
        self.recalculate_deltas(wrongly_predicted_target, correct_target)
        
        if self.num_hidden_layers > 0:
            
            # Loop through the target nodes and change the weights for the required targets
            for node in self.neural_network[-1]:
                
                # Only need to change the weights of the targets that should have been
                # predicted and weren't or that were wrongly predicted
                if node.target == correct_target or node.target == wrongly_predicted_target:
                    # Reassign the weights of the node
                    self.calc_weights(node, self.neural_network, -2)
            
            # Now loop through every layer between the output layer and the first hidden layer
            for layer_index, layer in enumerate(self.neural_network[1:-1]):
                for node in layer:
                    # Recalculate the node's weights
                    self.calc_weights(node, self.neural_network, layer_index)

        # recalculate the weights for the first hidden layer (or only layer if no hidden layers)
        # the data is the input this time, no previous nodes to get values from
        for node in self.neural_network[0]:
            self.calc_weights(node, data)

    # recalculate the weights of a particular node.  layer_index will determine which layer of the neural
    # network this node resides, unless it is on the first (or only) layer, in which case there will be no layer_index
    @staticmethod
    def calc_weights(node, values, prev_layer_index=None):
        # n used for calculating new weight
        n = -.1
        
        # this is the first layer in the network.
        if prev_layer_index is None:
            
            # loop through all the weights of vertices that are attached to input values
            for weight_index in range(len(node.input_weights) - 1):
                # reassign the node's weight
                node.input_weights[weight_index + 1] = node.input_weights[weight_index + 1] - (
                        n * node.delta * values[weight_index])
           
            # calculate the new weight for the bias node
            node.input_weights[0] = node.input_weights[0] - (n * node.delta * node.bias)
        
        else:  # this node resides in a hidden layer that isn't the first layer
           
            # loop through all the weights of vertices that are attached to input values
            for weight_index in range(len(node.input_weights) - 1):
                # reassign the node's weight
                node.input_weights[weight_index + 1] = node.input_weights[weight_index + 1] - (
                        n * node.delta * values[prev_layer_index][weight_index].value)
            
            # Calculate the new weight for the bias node
            node.input_weights[0] = node.input_weights[0] - (n * node.delta * node.bias)

    # recalculates the required deltas for the nodes
    def recalculate_deltas(self, wrongly_predicted_target, correct_target):
        
        # loop through each target node and recalculate if needed
        for target_node in self.neural_network[-1]:
            if target_node.target == correct_target:
               
                # calculate the error of the target node
                target_node.delta = target_node.value * (1 - target_node.value) * (target_node.value - 1)
            
            elif target_node.target == wrongly_predicted_target: 
                
                # calculate the error
                target_node.delta = target_node.value * (1 - target_node.value) * target_node.value
        
        # if there are more hidden layers, we're not done yet
        if self.num_hidden_layers > 0:
            
            # start at the hidden layers closest to the target and work backwards
            hidden_layer_index = len(self.neural_network) - 2
            
            # go through all the hidden layers 
            while hidden_layer_index >= 0:
                
                # loop through each node in this hidden layer
                for node_index, node in enumerate(self.neural_network[hidden_layer_index]):
                   
                    # calculate the sum for each next layer node's delta multiplied by the
                    # weight between that node and this node
                    sum_delta_weights = 0
                    for prev_node in self.neural_network[hidden_layer_index + 1]:
                        sum_delta_weights += prev_node.input_weights[node_index + 1] * prev_node.delta
                   
                    # calculate the new delta for the node
                    node.delta = node.value * (1 - node.value) * sum_delta_weights
                
                # decrement the index
                hidden_layer_index -= 1

    # gets the key of the item with the max value in a dictionary
    @staticmethod
    def get_key_with_max_value(dictionary):
        return max(dictionary.items(), key=operator.itemgetter(1))[0]

    # gets the accuracy of the current iteration
    def get_accuracy(self, data):
        num_predicted_correctly = 0
        for index, data_row in enumerate(data):
            if self.predict(data_row) == self.targets[index]:
                num_predicted_correctly += 1

        return num_predicted_correctly / len(self.targets)

    # predicts the target for a particular row of data
    def predict(self, data_row):
        
        if self.num_hidden_layers > 0:
            
            # 2D array to keep track of nodes values at each layer
            if self.num_hidden_layers > 1:
                hidden_node_values = [[] for _ in range(self.num_hidden_layers)]
            else:
                hidden_node_values = [[]]
           
            # set up the first layer as inputs
            for node in self.neural_network[0]:
                hidden_node_values[0].append(node.train(data_row))
            
            # set up all the hidden layers with the previous layer's activation as the value
            for layer_index, layer in enumerate(self.neural_network[1:-1]):
                for node in layer:
                    hidden_node_values[layer_index + 1].append(node.train(hidden_node_values[layer_index]))
            
            target_values = dict()
            
            # gets the activation for each target
            for node in self.neural_network[-1]:
                target_values[node.target] = node.train(hidden_node_values[-1])
            
            # Predicts the target with the highest activation value
            return min(target_values, key=target_values.get)#self.get_key_with_max_value(target_values)
        
        else:  # No hidden layers
            target_values = dict()
            
            # gets the activation for each target
            for node in self.neural_network[0]:
                target_values[node.target] = node.train(data_row)
            
            # predicts the target with the highest activation value
            return min(target_values, key=target_values.get)#self.get_key_with_max_value(target_values)


class NeuralNetworkModel:
    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.model = []

    def predict(self, data):
        for data_row in data:
            self.model.append(self.neural_network.predict(data_row))

        return self.model