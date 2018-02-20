from neuron import Neuron

#######################################
# NETWORK LAYER REQUIREMENTS 
# number neurons
# number attributes -> inputs/weights
# threshhold value
#######################################

class Network_Layer: 
    def __init__(self, num_neurons, num_attributes, threshhold):
        self.num_neurons = num_neurons
        self.neurons = []
        self.num_attributes = num_attributes + 1 # add 1 for bias
        self.threshhold = threshhold

    def create_layer(num_neurons):
        for i in range(0,num_neurons):
            self.neurons.append(Neuron(self.num_attributes,threshhold))

    def input_row(self, input):
        results = []
        data = list(inputs.copy())
        for i in self.neurons:
            results.append(i.input(data))
        return results

    def input(self, input_matrix):
        results = [] 
        for row in input_matrix:
            results.append(self.input_row(row))
        return results

    def __repr__(self):
        pass

       