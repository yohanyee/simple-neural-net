from neurons import *
from connections import *
import numpy as np

class NeuronLayer(object):
    def __init__(self, shape, neuron_activation='identity', neuron_bias=False):
        self.shape = shape
        self.size = np.prod(shape)
        self.neuron_activation = neuron_activation
        self.neuron_bias = neuron_bias
        self.neurons = None
        self.initialize_neurons()
        
    def initialize_neurons(self):
        neuron_array = np.empty(self.size, dtype=object)
        for element in range(self.size):
            neuron_array[element] = Neuron(activation_function=self.neuron_activation, bias=self.neuron_bias)
        self.neurons = neuron_array.reshape(self.shape)
        
    def reshape_neurons(self, shape):
        self.neurons = self.neurons.reshape(shape)
        self.shape = shape
        
    def reshape_to_3D(self):
        if len(self.shape)==2:
            self.reshape_neurons([1] + list(self.shape))
        
    def forward_pass(self):
        for neuron in self.neurons.reshape(self.size):
            neuron.forward_pass()
    
    def backward_pass(self):
        for neuron in self.neurons.reshape(self.size):
            neuron.backward_pass()
    
    def set_neuron_attributes(self, attribute, value_array):
        neurons = self.neurons.reshape(self.size)
        value_array = value_array.reshape(self.size)
        for element in range(self.size):
            setattr(neurons[element], attribute, value_array[element])  
                  
    def get_neuron_attributes(self, attribute):
        attribute_array = np.empty(self.size)
        neurons = self.neurons.reshape(self.size)
        for element in range(self.size):
            attrib = getattr(neurons[element], str(attribute))
            attribute_array[element] = attrib
        return(attribute_array.reshape(self.shape))
                 
    def activate(self, activation_array):
        self.set_neuron_attributes('a', activation_array)
    
    def get_activations(self):
        activation_array = self.get_neuron_attributes('a')
        return(activation_array)
        
    def get_activation_derivatives(self):
        activation_array = self.get_neuron_attributes('da')
        return(activation_array)
        
    def set_output_errors(self, wd_array):
        self.set_neuron_attributes('wd', wd_array)
        

class ConnectionLayer(object):
    def __init__(self, InputNeuronLayer, OutputNeuronLayer, connection_type=None):
        self.InputNeuronLayer = InputNeuronLayer
        self.OutputNeuronLayer = OutputNeuronLayer
        self.connections = []
        if connection_type is not None:
            self.set_connections(connection_type)
    
    def set_connections(self, connection_type):
        connection_type_functions = {   'full': self.full_connection    }
        connection_type_functions[connection_type]()
    
    def full_connection(self):
        input_neurons = self.InputNeuronLayer.neurons.reshape(self.InputNeuronLayer.size)
        output_neurons = self.OutputNeuronLayer.neurons.reshape(self.OutputNeuronLayer.size)
        for input_neuron in input_neurons:
            for output_neuron in output_neurons:
                self.add_connection(input_neuron, output_neuron)
      
    def add_connection(self, InputNeuronLayerNeuron, OutputNeuronLayerNeuron, Weight=None):
        self.connections.append(NeuronConnection(InputNeuronLayerNeuron, OutputNeuronLayerNeuron, Weight))
 
    def forward_pass(self):
        for connection in self.connections:
            connection.forward_pass()
        self.OutputNeuronLayer.forward_pass()
    
    def backward_pass(self):
        for connection in self.connections:
            connection.backward_pass()
        self.InputNeuronLayer.backward_pass()

                
class ConvolutionConnectionLayer(ConnectionLayer):
    def __init__(self, InputNeuronLayer, OutputNeuronLayer):
        super(ConvolutionConnectionLayer, self).__init__(InputNeuronLayer, OutputNeuronLayer, connection_type=None)
        self.InputNeuronLayer.reshape_to_3D()
        self.OutputNeuronLayer.reshape_to_3D() 
        self.kernel_shape = [InputNeuronLayer.shape[0], InputNeuronLayer.shape[1] - OutputNeuronLayer.shape[1] + 1, \
                                InputNeuronLayer.shape[2] - OutputNeuronLayer.shape[2] + 1 ]
        self.kernel_size = np.prod(self.kernel_shape)
        self.num_kernels = self.OutputNeuronLayer.shape[0]
        self.set_kernels()
        self.kernel_connection()
        
    def set_kernels(self):
        self.kernels = []
        for k in range(self.num_kernels):
            self.kernels.append(self.get_kernel())
    
    def get_kernel(self):
        return(ConvolutionKernel(self.kernel_shape))
        
    def kernel_connection(self):
        for k in range(len(self.kernels)):
            kernel = self.kernels[k]
            for x in range(self.OutputNeuronLayer.shape[1]):
                 for y in range(self.OutputNeuronLayer.shape[2]):
                     input_neuron_patch = self.InputNeuronLayer.neurons[ :, \
                        x:x+self.kernel_shape[1], y:y+self.kernel_shape[2]]
                     output_neuron = self.OutputNeuronLayer.neurons[k, x, y]
                     self.patch_connection(input_neuron_patch, output_neuron, kernel)
        
    def patch_connection(self, input_patch, output_neuron, kernel):
        for d in range(kernel.shape[0]):
            for a in range(kernel.shape[1]):
                for b in range(kernel.shape[2]):
                    input_neuron = input_patch[d,a,b]
                    kernel_weight = kernel.weights[d,a,b]
                    self.add_connection(input_neuron, output_neuron, kernel_weight)
                    
class MaxPoolingConnectionLayer(ConnectionLayer):
    def __init__(self, InputNeuronLayer, OutputNeuronLayer):
        super(MaxPoolingConnectionLayer, self).__init__(InputNeuronLayer, OutputNeuronLayer, connection_type=None)
        self.InputNeuronLayer.reshape_to_3D()
        self.OutputNeuronLayer.reshape_to_3D() 
        self.kernel_shape = [InputNeuronLayer.shape[1]/OutputNeuronLayer.shape[1], \
            InputNeuronLayer.shape[2]/OutputNeuronLayer.shape[2]]
        self.kernel_size = np.prod(self.kernel_shape)
        self.num_kernels = OutputNeuronLayer.size
        self.set_kernels()
        self.kernel_connection()
        
    def set_kernels(self):
        self.kernels = []
        for k in range(self.num_kernels):
            self.kernels.append(self.get_kernel())
        self.kernels = np.array(self.kernels).reshape(self.OutputNeuronLayer.shape)
    
    def get_kernel(self):
        return(MaxPoolingKernel(self.kernel_shape))
        
    def kernel_connection(self):
        for d in range(self.OutputNeuronLayer.shape[0]):
            for x in range(self.OutputNeuronLayer.shape[1]):
                for y in range(self.OutputNeuronLayer.shape[2]):
                    kernel = self.kernels[d, x, y]
                    kx = kernel.shape[0]
                    ky = kernel.shape[1]
                    input_neuron_patch = self.InputNeuronLayer.neurons[d, kx*x:kx*(x+1), ky*y:ky*(y+1)]
                    output_neuron = self.OutputNeuronLayer.neurons[d, x, y]
                    self.patch_connection(input_neuron_patch, output_neuron, kernel)
                
    def patch_connection(self, input_patch, output_neuron, kernel):
            for a in range(kernel.shape[0]):
                for b in range(kernel.shape[1]):
                    input_neuron = input_patch[a,b]
                    kernel_weight = kernel.weights[a,b]
                    self.add_connection(input_neuron, output_neuron, kernel_weight)
                    
    def forward_pass(self):
        activation_array = self.InputNeuronLayer.get_activations()
        for d in range(self.OutputNeuronLayer.shape[0]):
            for x in range(self.OutputNeuronLayer.shape[1]):
                for y in range(self.OutputNeuronLayer.shape[2]):
                    kx = self.kernel_shape[0]
                    ky = self.kernel_shape[1]
                    input_array = activation_array[d, kx*x:kx*(x+1), ky*y:ky*(y+1)]
                    self.kernels[d,x,y].set_switch(input_array)
        for connection in self.connections:
            connection.forward_pass()
        self.OutputNeuronLayer.forward_pass()
    
    def backward_pass(self):
        for connection in self.connections:
            connection.backward_pass()
        self.InputNeuronLayer.backward_pass()
        for kernel in self.kernels.reshape(self.kernels.size):
            kernel.clear_switch()
