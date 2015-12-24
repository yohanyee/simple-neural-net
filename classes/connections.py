from neurons import *
import numpy as np

class Weight(object):
    def __init__(self):
        self.weight = np.random.normal()
        self.weight_queue = 0.
        self.num_queue = 0
        self.trainable = True
    
    def queue(self, dCdw):
        self.weight_queue += dCdw
        self.num_queue += 1
        
    def update(self, lr):
        self.weight = self.weight - lr * self.weight_queue / self.num_queue
        self.weight_queue = 0.
        self.num_queue = 0
        
    def do_nothing(self, *args):
        pass

class BiasWeight(Weight):
    def __init__(self, bias=True):
        super(BiasWeight, self).__init__()
        self.trainable = bias
        if not bias:
            self.weight = 0.
            self.queue = self.do_nothing
            self.update = self.do_nothing

class ConvolutionWeight(Weight):
    def __init__(self, bias=True):
        super(ConvolutionWeight, self).__init__()
        self.num_queue = 1
        
    def queue(self, dCdw):
        self.weight_queue += dCdw       
        
    def update(self, lr):
        self.weight = self.weight - lr * self.weight_queue / self.num_queue
        self.weight_queue = 0.
 
class MaxPoolingWeight(Weight):
    def __init__(self):
        super(MaxPoolingWeight, self).__init__()
        self.weight = 0.
        self.trainable = False
        self.queue = self.do_nothing
        self.update = self.do_nothing
    
    def switch_on(self):
        self.weight = 1.
    
    def switch_off(self):
        self.weight = 0.

class NeuronConnection(object):
    def __init__(self, InputNeuron, OutputNeuron, WeightObject=None):
        self.InputNeuron = InputNeuron
        self.OutputNeuron = OutputNeuron
        if WeightObject is None:
            self.Weight = Weight()
        else:
            self.Weight = WeightObject
        
    def forward_pass(self):
        wx = self.InputNeuron.a * self.Weight.weight
        self.OutputNeuron.queue_forward_input(wx)
        
    def backward_pass(self):
        wd = self.OutputNeuron.error * self.Weight.weight
        self.InputNeuron.queue_backward_input(wd)
        self.Weight.queue(self.InputNeuron.a * self.OutputNeuron.error)

class Kernel(object):
    def __init__(self, shape):
        self.shape = shape
        self.size = np.prod(shape)
        self.weights = None
                
    def initialize_kernel(self):
        weight_array = np.empty(self.size, dtype=object)
        for element in range(self.size):
            weight_array[element] = Weight()
        self.weights = weight_array.reshape(self.shape)
        
    def get_weights(self):
        weight_array = np.array([ Weight.weight for Weight in self.weights.reshape(self.size) ])
        return(weight_array.reshape(self.shape))

class ConvolutionKernel(Kernel):
    def __init__(self, shape):
        super(ConvolutionKernel, self).__init__(shape)
        self.initialize_kernel()
        
    def initialize_kernel(self):
        weight_array = np.empty(self.size, dtype=object)
        for element in range(self.size):
            weight_array[element] = ConvolutionWeight()
        self.weights = weight_array.reshape(self.shape)

class MaxPoolingKernel(Kernel):
    def __init__(self, shape):
        super(MaxPoolingKernel, self).__init__(shape)
        self.initialize_kernel()
        
    def initialize_kernel(self):
        weight_array = np.empty(self.size, dtype=object)
        for element in range(self.size):
            weight_array[element] = MaxPoolingWeight()
        self.weights = weight_array.reshape(self.shape)
    
    def set_switch(self, input_array):
        self.switch_index = np.argmax(input_array)
        self.weights.reshape(self.size)[self.switch_index].switch_on()
    
    def clear_switch(self): 
        self.weights.reshape(self.size)[self.switch_index].switch_off()
