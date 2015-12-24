import numpy as np
from data import *
from construct import *
from train import *
from hippocampi_to_patches import *
    
class DigitsPipeline(object):
    def __init__(self):
        self.D = Data()
        self.D.load_digits_data()
        self.D.reshape([16,16],[1])
    
        self.N = FeedForwardNetwork()
        self.N.auto_add_layer_fullyconnected(NeuronLayer(self.D.input_shape, neuron_activation='identity'))
        self.N.auto_add_layer_fullyconnected(NeuronLayer(10, neuron_activation='logistic', neuron_bias=True))
        self.N.auto_add_layer_fullyconnected(NeuronLayer(self.D.output_shape, neuron_activation='logistic', neuron_bias=True))
        
        self.Trainer = BackpropagationTrainer(self.D, self.N)

class DigitsConvolutionPipeline(object):
    def __init__(self):
        self.D = Data()
        self.D.load_digits_data()
        self.D.reshape([16,16],[1])
    
        self.N = FeedForwardNetwork()
        self.N.auto_add_layer_fullyconnected(NeuronLayer(self.D.input_shape, neuron_activation='identity'))
        self.N.auto_add_layer_convolution(NeuronLayer([4,8,8], neuron_activation='logistic', neuron_bias=True))
        self.N.auto_add_layer_fullyconnected(NeuronLayer(10, neuron_activation='logistic', neuron_bias=True))
        self.N.auto_add_layer_fullyconnected(NeuronLayer(self.D.output_shape, neuron_activation='logistic', neuron_bias=True))
        
        self.Trainer = BackpropagationTrainer(self.D, self.N)
       
        
