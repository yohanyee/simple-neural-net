import numpy as np

class NonlinearActivationFunctions(object):
    def __init__(self):
        self.functions = {  "identity" : self.identity, \
                        "relu"     : self.relu, \
                        "logistic" : self.logistic }
        self.gradients = {  "identity" : self.d_identity, \
                        "relu"     : self.d_relu, \
                        "logistic" : self.d_logistic }

    def identity(self, z):
        return(z)

    def relu(self, z):
        return(max(0., z))
        
    def logistic(self, z):
        return(1./(1. + np.exp(-z)))
        
    def d_identity(self, z):
        return(1.)
        
    def d_relu(self, z):
        if z <= 0.:
            d = 0.
        else:
            d = 1.
        return(d)
    
    def d_logistic(self, z):
        d = self.logistic(z)*(1.-self.logistic(z))
        return(d)
