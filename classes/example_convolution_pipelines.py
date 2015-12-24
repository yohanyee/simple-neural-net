import numpy as np
from data import *
from construct import *
from train import *

class TestPipeline(object):
    def __init__(self):
        self.D = Data()
        self.D.load_digits_data()
        self.D.reshape([16,16],[1])
    
        self.N = FeedForwardNetwork()
        self.N.auto_add_layer_fullyconnected(NeuronLayer(self.D.input_shape, neuron_activation='identity'))
        self.N.auto_add_layer_convolution(NeuronLayer([10,12,12], neuron_activation='logistic', neuron_bias=True))
        self.N.auto_add_layer_maxpooling(NeuronLayer([10,6,6], neuron_activation='logistic', neuron_bias=False))
        self.N.auto_add_layer_fullyconnected(NeuronLayer(50, neuron_activation='logistic', neuron_bias=True))
        self.N.auto_add_layer_fullyconnected(NeuronLayer(self.D.output_shape, neuron_activation='logistic', neuron_bias=True))
        
        self.Trainer = BackpropagationTrainer(self.D, self.N)
        
class ClassificationPipeline(object):
    # THIS ONE IS GOOD
    def __init__(self):
        self.D = Data()
        self.D.load_digits_data()
        self.D.reshape([16,16],[1])
    
        self.N = FeedForwardNetwork()
        self.N.auto_add_layer_fullyconnected(NeuronLayer(self.D.input_shape, neuron_activation='identity'))
        self.N.auto_add_layer_convolution(NeuronLayer([10,12,12], neuron_activation='logistic', neuron_bias=True))
        self.N.auto_add_layer_convolution(NeuronLayer([10,8,8], neuron_activation='logistic', neuron_bias=False))
        self.N.auto_add_layer_convolution(NeuronLayer([10,4,4], neuron_activation='logistic', neuron_bias=False))
        self.N.auto_add_layer_convolution(NeuronLayer([10,2,2], neuron_activation='logistic', neuron_bias=False))
        self.N.auto_add_layer_fullyconnected(NeuronLayer(50, neuron_activation='logistic', neuron_bias=True))
        self.N.auto_add_layer_fullyconnected(NeuronLayer(self.D.output_shape, neuron_activation='logistic', neuron_bias=False))
        
        self.Trainer = BackpropagationTrainer(self.D, self.N)
        
class ClassificationPipelineWithMaxPooling(object):
    # THIS ONE IS ALSO GOOD
    def __init__(self):
        self.D = Data()
        self.D.load_digits_data()
        self.D.reshape([16,16],[1])
    
        self.N = FeedForwardNetwork()
        self.N.auto_add_layer_fullyconnected(NeuronLayer(self.D.input_shape, neuron_activation='identity'))
        self.N.auto_add_layer_convolution(NeuronLayer([10,12,12], neuron_activation='logistic', neuron_bias=True))
        self.N.auto_add_layer_maxpooling(NeuronLayer([10,6,6], neuron_activation='identity', neuron_bias=False))
        self.N.auto_add_layer_convolution(NeuronLayer([10,2,2], neuron_activation='logistic', neuron_bias=False))
        self.N.auto_add_layer_maxpooling(NeuronLayer([10,1,1], neuron_activation='identity', neuron_bias=False))
        self.N.auto_add_layer_fullyconnected(NeuronLayer(50, neuron_activation='logistic', neuron_bias=True))
        self.N.auto_add_layer_fullyconnected(NeuronLayer(self.D.output_shape, neuron_activation='logistic', neuron_bias=False))
        
        self.Trainer = BackpropagationTrainer(self.D, self.N)
