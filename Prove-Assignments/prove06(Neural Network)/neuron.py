import random

#######################################
# NEURON REQUIREMENTS 
# number inputs
# number weights
# threshhold value
#######################################

class Neuron: 
    def __init__(self, num_weights, threshhold):
        self.num_weights = num_weights
        self.weights = []
        self.threshhold = threshhold

        # store the weight value in an array
        # make a changable float value as 
        # weight values will need to be changed
        for i in range(0,num_weights):
            self.weights.append(random.uniform(-1.0, 1.0))

    def input(self, neuron_input): 
        if (self.num_weights != neuron_input):
            print("ERROR! Number of weights do not match number of neuron inputs")
        else:
            # multiply values && consider threshhold
            synapse_value = 0
            threshhold = 0
            for i in range (0, self.num_weights):
                synapse_value = self.weights[i] * neuron_input[i]
            if (synapse_value >= self.threshhold):
                return 1
            else: 
                return 0
    
    def __repr__(self):
        return 'Neuron({},{})'.format(self.num_weights, self.weights)

