import numpy as np
from functions import *
from connections import *

class Neuron(object):
    def __init__(self, activation_function='identity', bias=False):
        self.activation_function = NonlinearActivationFunctions().functions[activation_function]
        self.activation_derivative = NonlinearActivationFunctions().gradients[activation_function]
        self.Bias = BiasWeight(bias=bias)
        self.b = self.Bias.weight
        self.wx = np.random.random()
        self.z = self.wx + self.b
        self.a = self.activation_function(self.z)
        self.da = self.activation_derivative(self.z)
        self.wd = np.random.random()
        self.error = self.da * self.wd
    
    def queue_forward_input(self, wx):
        self.wx += wx

    def forward_pass(self):
        self.z = self.wx + self.b
        self.a = self.activation_function(self.z)
        self.da = self.activation_derivative(self.z)
        self.wx = 0.
        
    def queue_backward_input(self, wd):
        self.wd += wd
    
    def backward_pass(self):
        self.error = self.da * self.wd
        self.Bias.queue(self.error)
        self.wd = 0.
    
    def activate(self, a):
        self.a = a
        
