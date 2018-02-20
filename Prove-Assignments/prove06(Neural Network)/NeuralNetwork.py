from neuron import Neuron
from network_layer import Network_Layer

def main(): 
    neuron1 = Neuron(2)

    print("testing neuron")
    neuron1.input(0.5)
    print ("neuron1 " + neuron1)


if __name__ == "__main__":
    main()