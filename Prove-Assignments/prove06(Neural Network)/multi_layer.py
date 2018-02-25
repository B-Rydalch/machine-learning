import pandas as pd
import numpy as np
from neuron import Neuron
from network_layer import Network_Layer

class Multi_Network_layer:
   def __init__(self, num_attributes, num_neuron_per_layer = 3): 
      self.layers = []
      input_size = num_attributes
      if (type(num_neuron_per_layer) is list):
         for counter in num_neuron_per_layer:
            self.layers.append(
               SingleLayerPerceptron(
                  num_attributes=input_size, 
                  num_neurons=counter 
               ))
            input_size = counter
      else:
         self.layers.append(
            SingleLayerPerceptron(
               num_attributes=input_size, 
               num_neurons=num_neuron_per_layer
            ))

   def input_row(self, input_):
      input = input_
      for layer in self.layers:
         input = layer.input_row(input)
      return input

   def input(self, input_matrix):
      results = []
      for row in input_matrix:
         results.append(self.input_one(row))
      return results
      

   def __repr__(self):
      reprr = ""
      for layer in self.layers:
         reprr += '{}>({})<{} '.format(layer.num_neurons)
      return reprr