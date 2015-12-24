import numpy as np
from layers import *
from connections import *
from neurons import *
from util import *
from train import *
from cost import *

class NetworkConstructor(object):
    def set_input_layer(self, NeuronLayer):
        self.InputLayer = NeuronLayer
        
    def set_output_layer(self, NeuronLayer):
        self.OutputLayer = NeuronLayer
    
    def add_neuron_layer(self, NeuronLayer):
        self.NeuronLayers.append(NeuronLayer)
        for neuron in NeuronLayer.neurons.reshape(NeuronLayer.size):
            if neuron.Bias.trainable:
                self.Weights.append(neuron.Bias)
    
    def add_connection_layer(self, ConnectionLayer):
        self.ConnectionLayers.append(ConnectionLayer)
        for connection in ConnectionLayer.connections:
            if connection.Weight not in self.Weights:
                if connection.Weight.trainable:
                    self.Weights.append(connection.Weight)
       
    def auto_add_layer_fullyconnected(self, NeuronLayer):
        self.add_neuron_layer(NeuronLayer)
        if len(self.NeuronLayers)==1:
            self.set_input_layer(NeuronLayer)
        else:
            self.add_connection_layer(ConnectionLayer(self.NeuronLayers[-2], \
                self.NeuronLayers[-1], connection_type='full'))
            self.set_output_layer(NeuronLayer)
            
    def auto_add_layer_convolution(self, NeuronLayer):
        self.add_neuron_layer(NeuronLayer)
        if len(self.NeuronLayers)==1:
            self.set_input_layer(NeuronLayer)
        else:
            self.add_connection_layer(ConvolutionConnectionLayer(self.NeuronLayers[-2], \
                self.NeuronLayers[-1]))
            self.set_output_layer(NeuronLayer)
    
    def auto_add_layer_maxpooling(self, NeuronLayer):
        self.add_neuron_layer(NeuronLayer)
        if len(self.NeuronLayers)==1:
            self.set_input_layer(NeuronLayer)
        else:
            self.add_connection_layer(MaxPoolingConnectionLayer(self.NeuronLayers[-2], \
                self.NeuronLayers[-1]))
            self.set_output_layer(NeuronLayer)

class FeedForwardNetwork(NetworkConstructor):
    def __init__(self):
        self.NeuronLayers = []
        self.ConnectionLayers = []
        self.Weights = []
        self.InputLayer = None
        self.OutputLayer = None
        self.trained = False
        self.Cost = Cost()
        
    def infer(self, input_array):
        self.InputLayer.activate(input_array)
        for connection_layer in self.ConnectionLayers:
            connection_layer.forward_pass()
        output_array = self.OutputLayer.get_activations()
        return(output_array)
        
    def backpropagate_errors(self, true_output_array):
        cost_wd = self.Cost.cost_derivative(true_output_array, self.OutputLayer.get_activations())
        self.OutputLayer.set_output_errors(cost_wd)
        self.OutputLayer.backward_pass()
        for connection_layer in list(reversed(self.ConnectionLayers)):
            connection_layer.backward_pass()
        
    def update(self, lr):
        for weight in self.Weights:
            weight.update(lr)

    def study(self, input_array, true_output_array):
        self.infer(input_array)
        self.backpropagate_errors(true_output_array)
        
    def learn(self, input_array, true_output_array, lr=1.):
        self.study(input_array, true_output_array)
        self.update(lr)
        
class DoubleFeedForwardNetwork(NetworkConstructor):
    def __init__(self):
        self.NeuronLayers = []
        self.ConnectionLayers = []
        self.Weights = []
        self.InputLayer1 = None
        self.InputLayer2 = None
        self.OutputLayer = None
        self.trained = False
        self.Cost = Cost()
        
    def infer(self, input_array):
        input_array1 = input_array[0]
        input_array2 = input_array[1]
        self.InputLayer1.activate(input_array1)
        self.InputLayer2.activate(input_array2)
        for connection_layer in self.ConnectionLayers:
            connection_layer.forward_pass()
        output_array = self.OutputLayer.get_activations()
        return(output_array)
        
    def backpropagate_errors(self, true_output_array):
        cost_wd = self.Cost.cost_derivative(true_output_array, self.OutputLayer.get_activations())
        self.OutputLayer.set_output_errors(cost_wd)
        self.OutputLayer.backward_pass()
        for connection_layer in list(reversed(self.ConnectionLayers)):
            connection_layer.backward_pass()
        
    def update(self, lr):
        for weight in self.Weights:
            weight.update(lr)

    def study(self, input_array, true_output_array):
        self.infer(input_array)
        self.backpropagate_errors(true_output_array)
        
    def learn(self, input_array, true_output_array, lr=1.):
        self.study(input_array, true_output_array)
        self.update(lr)        
