import numpy as np

class Cost(object):
    def __init__(self, function='square_error'):
        self.error = np.inf
        self.min_error = np.inf
        self.cost_function = getattr(self, function)
        self.cost_derivative = getattr(self, 'd_'+function)
        
    def square_error(self, true_output_array, inferred_output_array):
        error = 0.5*np.sum((inferred_output_array - true_output_array)**2)
        self.error = error
        self.min_error = min(self.min_error, error)
        return(error)
        
    def d_square_error(self, true_output_array, inferred_output_array):
        d_error = inferred_output_array - true_output_array
        return(d_error)
